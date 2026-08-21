#include <windows.h>
#include <windows.graphics.capture.interop.h>
#include <windows.graphics.directx.direct3d11.interop.h>
#include <wincodec.h>

#include <d3d11_4.h>
#include <dxgi1_6.h>
#include <shellapi.h>

#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Graphics.Capture.h>
#include <winrt/Windows.Graphics.DirectX.Direct3D11.h>
#include <winrt/Windows.Graphics.DirectX.h>
#include <winrt/Windows.Graphics.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

using namespace std;

namespace wgc = winrt::Windows::Graphics::Capture;
namespace wg = winrt::Windows::Graphics;
namespace wgd = winrt::Windows::Graphics::DirectX;
namespace wgd11 = winrt::Windows::Graphics::DirectX::Direct3D11;

namespace {
constexpr wchar_t kWindowClassName[] = L"AvionCaptureHostWindow";
constexpr UINT kUpdateFpsMessage = WM_APP + 1;
constexpr LONGLONG kHundredNanosecondsPerSecond = 10'000'000;
constexpr LONGLONG kOneSecondTimestamp = kHundredNanosecondsPerSecond;

template <typename T>
winrt::com_ptr<T> GetDXGIInterfaceFromObject(winrt::Windows::Foundation::IInspectable const& object) {
    auto access = object.as<::Windows::Graphics::DirectX::Direct3D11::IDirect3DDxgiInterfaceAccess>();
    winrt::com_ptr<T> result;
    winrt::check_hresult(access->GetInterface(winrt::guid_of<T>(), result.put_void()));
    return result;
}

wstring HResultMessage(winrt::hresult_error const& error) {
    wstringstream stream;
    stream << L"0x" << hex << static_cast<uint32_t>(error.code()) << L": " << error.message().c_str();
    return stream.str();
}

void ShowError(wstring const& message) {
    MessageBoxW(nullptr, message.c_str(), L"Avion Capture Host", MB_OK | MB_ICONERROR);
}

wstring HResultMessage(HRESULT hr) {
    return HResultMessage(winrt::hresult_error(hr));
}

wstring ToWide(char const* value) {
    return winrt::to_hstring(value ? value : "").c_str();
}

wstring TimestampFileName() {
    SYSTEMTIME now{};
    GetLocalTime(&now);

    wchar_t buffer[32]{};
    swprintf_s(
        buffer,
        L"%04u%02u%02u-%02u%02u%02u",
        now.wYear,
        now.wMonth,
        now.wDay,
        now.wHour,
        now.wMinute,
        now.wSecond);
    return buffer;
}

enum class RecordingMode {
    RawBgra,
    PngOnePerSecond,
};

wstring ZeroPaddedNumber(uint32_t value, int width) {
    wstringstream stream;
    stream << setw(width) << setfill(L'0') << value;
    return stream.str();
}

bool PathExists(filesystem::path const& path) {
    error_code error;
    return filesystem::exists(path, error);
}

filesystem::path RecordingPathForBaseName(filesystem::path const& recordingDir, wstring const& baseName, RecordingMode mode) {
    if (mode == RecordingMode::PngOnePerSecond) {
        return recordingDir / (baseName + L"-1fps");
    }

    return recordingDir / (baseName + L".bgra");
}

filesystem::path DefaultRecordingPath(RecordingMode mode) {
    const filesystem::path recordingDir = filesystem::current_path() / L"recordings";
    const wstring baseName = L"avion-capture-" + TimestampFileName();

    for (uint32_t attempt = 0; attempt < 1000; ++attempt) {
        wstring uniqueBaseName = baseName;
        if (attempt > 0) {
            uniqueBaseName += L"-" + ZeroPaddedNumber(attempt + 1, 3);
        }

        filesystem::path candidate = RecordingPathForBaseName(recordingDir, uniqueBaseName, mode);
        if (!PathExists(candidate)) {
            return candidate;
        }
    }

    return RecordingPathForBaseName(recordingDir, baseName + L"-" + to_wstring(GetTickCount64()), mode);
}

struct HostOptions {
    bool showCursor{};
    bool recordVideo{};
    RecordingMode recordingMode{ RecordingMode::RawBgra };
    filesystem::path recordPath{};
};

bool TryGetInlineOptionValue(wstring const& argument, wstring const& name, wstring& value) {
    const wstring prefix = name + L"=";
    if (argument.rfind(prefix, 0) != 0) {
        return false;
    }

    value = argument.substr(prefix.size());
    return true;
}

RecordingMode ParseRecordingMode(wstring const& value) {
    if (value == L"png-1fps" || value == L"1fps" || value == L"one-fps") {
        return RecordingMode::PngOnePerSecond;
    }

    return RecordingMode::RawBgra;
}

HostOptions ParseHostOptions() {
    HostOptions options{};

    int argc = 0;
    PWSTR* argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    if (!argv) {
        return options;
    }

    for (int index = 1; index < argc; ++index) {
        const wstring argument = argv[index];
        wstring value;

        if (argument == L"--show-cursor") {
            options.showCursor = true;
        }
        else if (argument == L"--record-video" || argument == L"--record") {
            options.recordVideo = true;
        }
        else if (argument == L"--record-path" && index + 1 < argc) {
            options.recordVideo = true;
            options.recordPath = argv[++index];
        }
        else if (TryGetInlineOptionValue(argument, L"--record-path", value)) {
            options.recordVideo = true;
            options.recordPath = value;
        }
        else if (argument == L"--record-mode" && index + 1 < argc) {
            options.recordVideo = true;
            options.recordingMode = ParseRecordingMode(argv[++index]);
        }
        else if (TryGetInlineOptionValue(argument, L"--record-mode", value)) {
            options.recordVideo = true;
            options.recordingMode = ParseRecordingMode(value);
        }
    }

    LocalFree(argv);

    if (options.recordVideo && options.recordPath.empty()) {
        options.recordPath = DefaultRecordingPath(options.recordingMode);
    }

    return options;
}

class MappedTexture {
public:
    MappedTexture(ID3D11DeviceContext* context, ID3D11Texture2D* texture)
        : m_context(context),
        m_texture(texture) {
        winrt::check_hresult(m_context->Map(m_texture, 0, D3D11_MAP_READ, 0, &m_mapped));
    }

    ~MappedTexture() {
        if (m_context && m_texture) {
            m_context->Unmap(m_texture, 0);
        }
    }

    D3D11_MAPPED_SUBRESOURCE const& Data() const {
        return m_mapped;
    }

private:
    ID3D11DeviceContext* m_context{};
    ID3D11Texture2D* m_texture{};
    D3D11_MAPPED_SUBRESOURCE m_mapped{};
};

class LosslessFrameRecorder {
public:
    void Configure(bool enabled, RecordingMode mode, filesystem::path outputPath) {
        m_enabled = enabled;
        m_mode = mode;
        m_baseOutputPath = move(outputPath);
    }

    void Initialize(ID3D11Device* device, int width, int height) {
        if (!m_enabled || m_failed) {
            return;
        }

        try {
            Start(device, width, height);
        }
        catch (winrt::hresult_error const& error) {
            MarkFailed(L"Recording failed: " + HResultMessage(error));
        }
        catch (exception const& error) {
            MarkFailed(L"Recording failed: " + ToWide(error.what()));
        }
    }

    void Stop() {
        WriteMetadata();
        ResetWriter();
    }

    void WriteFrame(ID3D11DeviceContext* context, ID3D11Texture2D* sourceTexture) {
        if (!m_enabled || m_failed || !m_timingStream.is_open()) {
            return;
        }

        try {
            D3D11_TEXTURE2D_DESC sourceDescription{};
            sourceTexture->GetDesc(&sourceDescription);
            if (sourceDescription.Width != static_cast<UINT>(m_width) ||
                sourceDescription.Height != static_cast<UINT>(m_height)) {
                RestartForSize(context, sourceDescription.Width, sourceDescription.Height);
                if (!m_timingStream.is_open()) {
                    return;
                }
            }

            const LONGLONG timestamp = CurrentTimestamp();
            const LONGLONG delta = m_lastFrameTimestamp < 0 ? 0 : timestamp - m_lastFrameTimestamp;
            const bool shouldSavePixels = m_mode == RecordingMode::RawBgra || ShouldSaveSnapshot(timestamp);
            filesystem::path savedFile;

            if (shouldSavePixels) {
                context->CopyResource(m_stagingTexture.get(), sourceTexture);

                MappedTexture mapped(context, m_stagingTexture.get());
                CopyMappedFrame(mapped.Data());

                if (m_mode == RecordingMode::RawBgra) {
                    m_rawStream.write(reinterpret_cast<const char*>(m_frameBuffer.data()), m_frameBufferSize);
                    if (!m_rawStream.good()) {
                        throw runtime_error("Failed to write the lossless frame data.");
                    }
                    ++m_savedFrameCount;
                }
                else {
                    savedFile = SnapshotPath();
                    SavePng(savedFile);
                    m_lastSnapshotTimestamp = timestamp;
                    ++m_savedFrameCount;
                }
            }

            WriteFrameTiming(timestamp, delta, shouldSavePixels, savedFile);
            m_lastFrameTimestamp = timestamp;
            ++m_frameCount;
        }
        catch (winrt::hresult_error const& error) {
            MarkFailed(L"Recording failed: " + HResultMessage(error));
            ResetWriter();
        }
        catch (exception const& error) {
            MarkFailed(L"Recording failed: " + ToWide(error.what()));
            ResetWriter();
        }
    }

    bool IsEnabled() const {
        return m_enabled;
    }

    bool HasFailed() const {
        return m_failed;
    }

    wstring TitleSuffix() const {
        if (!m_enabled) {
            return L"";
        }

        if (m_failed) {
            return L" - REC failed";
        }

        if (m_timingStream.is_open()) {
            return m_mode == RecordingMode::PngOnePerSecond ? L" - REC 1 FPS PNG" : L" - REC lossless";
        }

        return L" - REC pending";
    }

private:
    void Start(ID3D11Device* device, int width, int height) {
        width = max(width, 1);
        height = max(height, 1);

        Stop();

        m_width = width;
        m_height = height;
        m_frameStride = static_cast<size_t>(m_width) * 4;

        const uint64_t frameBufferSize = static_cast<uint64_t>(m_frameStride) * static_cast<uint64_t>(m_height);
        if (frameBufferSize > static_cast<uint64_t>(numeric_limits<size_t>::max())) {
            throw runtime_error("The captured frame is too large to record.");
        }
        m_frameBufferSize = static_cast<size_t>(frameBufferSize);
        m_frameBuffer.resize(m_frameBufferSize);

        D3D11_TEXTURE2D_DESC stagingDescription{};
        stagingDescription.Width = static_cast<UINT>(m_width);
        stagingDescription.Height = static_cast<UINT>(m_height);
        stagingDescription.MipLevels = 1;
        stagingDescription.ArraySize = 1;
        stagingDescription.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        stagingDescription.SampleDesc.Count = 1;
        stagingDescription.Usage = D3D11_USAGE_STAGING;
        stagingDescription.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        winrt::check_hresult(device->CreateTexture2D(&stagingDescription, nullptr, m_stagingTexture.put()));

        m_outputPath = OutputPathForCurrentPart();

        if (m_mode == RecordingMode::PngOnePerSecond) {
            filesystem::create_directories(m_outputPath);
            m_timingPath = m_outputPath / L"frames.csv";
            m_metadataPath = m_outputPath / L"capture.json";
        }
        else {
            if (!m_outputPath.parent_path().empty()) {
                filesystem::create_directories(m_outputPath.parent_path());
            }
            m_timingPath = SidecarPath(L".csv");
            m_metadataPath = SidecarPath(L".json");

            m_rawStream.open(m_outputPath, ios::binary | ios::trunc);
            if (!m_rawStream) {
                throw runtime_error("Failed to open the lossless frame data file.");
            }
        }

        m_timingStream.open(m_timingPath, ios::trunc);
        if (!m_timingStream) {
            throw runtime_error("Failed to open the frame timing metadata file.");
        }

        m_timingStream << "frame_index,timestamp_100ns,timestamp_seconds,delta_100ns,instant_fps,saved,output_file\n";
        m_frameCount = 0;
        m_savedFrameCount = 0;
        m_startTime = {};
        m_lastFrameTimestamp = -1;
        m_lastSnapshotTimestamp = -1;
        WriteMetadata();
    }

    void RestartForSize(ID3D11DeviceContext* context, UINT width, UINT height) {
        winrt::com_ptr<ID3D11Device> device;
        context->GetDevice(device.put());

        ++m_partIndex;
        Start(device.get(), static_cast<int>(width), static_cast<int>(height));
    }

    filesystem::path OutputPathForCurrentPart() const {
        if (m_partIndex == 0) {
            return m_baseOutputPath;
        }

        const wstring partSuffix = L"-part" + to_wstring(m_partIndex + 1);
        if (m_mode == RecordingMode::PngOnePerSecond) {
            return m_baseOutputPath.parent_path() / (m_baseOutputPath.filename().wstring() + partSuffix);
        }

        return m_baseOutputPath.parent_path() /
            (m_baseOutputPath.stem().wstring() + partSuffix + m_baseOutputPath.extension().wstring());
    }

    filesystem::path SidecarPath(wstring const& extension) const {
        return m_outputPath.parent_path() / (m_outputPath.stem().wstring() + extension);
    }

    LONGLONG CurrentTimestamp() {
        const auto now = chrono::steady_clock::now();
        if (m_startTime.time_since_epoch().count() == 0) {
            m_startTime = now;
            return 0;
        }

        const auto elapsed = now - m_startTime;
        return chrono::duration_cast<chrono::duration<LONGLONG, ratio<1, kHundredNanosecondsPerSecond>>>(elapsed).count();
    }

    bool ShouldSaveSnapshot(LONGLONG timestamp) const {
        return m_savedFrameCount == 0 || timestamp - m_lastSnapshotTimestamp >= kOneSecondTimestamp;
    }

    filesystem::path SnapshotPath() const {
        wchar_t fileName[32]{};
        swprintf_s(fileName, L"frame_%06llu.png", static_cast<unsigned long long>(m_savedFrameCount + 1));
        return m_outputPath / fileName;
    }

    void CopyMappedFrame(D3D11_MAPPED_SUBRESOURCE const& mappedData) {
        const auto* source = static_cast<const BYTE*>(mappedData.pData);
        for (int row = 0; row < m_height; ++row) {
            memcpy(
                m_frameBuffer.data() + (static_cast<size_t>(row) * m_frameStride),
                source + (static_cast<size_t>(row) * mappedData.RowPitch),
                m_frameStride);
        }
    }

    void SavePng(filesystem::path const& outputPath) {
        if (m_frameStride > numeric_limits<UINT>::max() ||
            m_frameBufferSize > numeric_limits<UINT>::max()) {
            throw runtime_error("The captured frame is too large to encode as PNG.");
        }

        winrt::com_ptr<IWICImagingFactory> factory;
        winrt::check_hresult(CoCreateInstance(
            CLSID_WICImagingFactory2,
            nullptr,
            CLSCTX_INPROC_SERVER,
            __uuidof(IWICImagingFactory),
            factory.put_void()));

        winrt::com_ptr<IWICStream> stream;
        winrt::check_hresult(factory->CreateStream(stream.put()));
        winrt::check_hresult(stream->InitializeFromFilename(outputPath.wstring().c_str(), GENERIC_WRITE));

        winrt::com_ptr<IWICBitmapEncoder> encoder;
        winrt::check_hresult(factory->CreateEncoder(GUID_ContainerFormatPng, nullptr, encoder.put()));
        winrt::check_hresult(encoder->Initialize(stream.get(), WICBitmapEncoderNoCache));

        winrt::com_ptr<IWICBitmapFrameEncode> frame;
        winrt::check_hresult(encoder->CreateNewFrame(frame.put(), nullptr));
        winrt::check_hresult(frame->Initialize(nullptr));
        winrt::check_hresult(frame->SetSize(static_cast<UINT>(m_width), static_cast<UINT>(m_height)));

        WICPixelFormatGUID format = GUID_WICPixelFormat32bppBGRA;
        winrt::check_hresult(frame->SetPixelFormat(&format));
        if (!IsEqualGUID(format, GUID_WICPixelFormat32bppBGRA)) {
            throw runtime_error("WIC could not preserve the BGRA pixel format.");
        }

        winrt::check_hresult(frame->WritePixels(
            static_cast<UINT>(m_height),
            static_cast<UINT>(m_frameStride),
            static_cast<UINT>(m_frameBufferSize),
            m_frameBuffer.data()));
        winrt::check_hresult(frame->Commit());
        winrt::check_hresult(encoder->Commit());
    }

    void WriteFrameTiming(LONGLONG timestamp, LONGLONG delta, bool saved, filesystem::path const& savedFile) {
        if (!m_timingStream.is_open()) {
            return;
        }

        const double seconds = static_cast<double>(timestamp) / static_cast<double>(kHundredNanosecondsPerSecond);
        const double instantFps = delta > 0
            ? static_cast<double>(kHundredNanosecondsPerSecond) / static_cast<double>(delta)
            : 0.0;

        m_timingStream
            << m_frameCount << ","
            << timestamp << ","
            << fixed << setprecision(7) << seconds << ","
            << delta << ","
            << fixed << setprecision(3) << instantFps << ","
            << (saved ? 1 : 0) << ","
            << (savedFile.empty() ? "" : savedFile.filename().string()) << "\n";
    }

    void WriteMetadata() {
        if (!m_enabled || m_outputPath.empty()) {
            return;
        }

        ofstream metadata(m_metadataPath, ios::trunc);
        if (!metadata) {
            return;
        }

        const double durationSeconds = m_lastFrameTimestamp > 0
            ? static_cast<double>(m_lastFrameTimestamp) / static_cast<double>(kHundredNanosecondsPerSecond)
            : 0.0;
        const double averageFps = durationSeconds > 0.0
            ? static_cast<double>(m_frameCount) / durationSeconds
            : 0.0;

        metadata
            << "{\n"
            << "  \"format\": \"" << (m_mode == RecordingMode::PngOnePerSecond ? "avion.lossless.png-1fps" : "avion.lossless.bgra") << "\",\n"
            << "  \"captureMode\": \"" << (m_mode == RecordingMode::PngOnePerSecond ? "png-1fps" : "raw-bgra") << "\",\n";

        if (m_mode == RecordingMode::PngOnePerSecond) {
            metadata << "  \"frameDirectory\": \"" << m_outputPath.filename().string() << "\",\n";
        }
        else {
            metadata << "  \"dataFile\": \"" << m_outputPath.filename().string() << "\",\n";
        }

        metadata
            << "  \"timingFile\": \"" << m_timingPath.filename().string() << "\",\n"
            << "  \"width\": " << m_width << ",\n"
            << "  \"height\": " << m_height << ",\n"
            << "  \"pixelFormat\": \"bgra\",\n"
            << "  \"rowOrder\": \"top-down\",\n"
            << "  \"bytesPerPixel\": 4,\n"
            << "  \"frameStrideBytes\": " << m_frameStride << ",\n"
            << "  \"frameSizeBytes\": " << m_frameBufferSize << ",\n"
            << "  \"observedFrameCount\": " << m_frameCount << ",\n"
            << "  \"savedFrameCount\": " << m_savedFrameCount << ",\n"
            << "  \"timebase\": \"100ns\",\n"
            << "  \"durationSeconds\": " << fixed << setprecision(7) << durationSeconds << ",\n"
            << "  \"averageObservedFps\": " << fixed << setprecision(3) << averageFps << ",\n"
            << "  \"lossless\": true\n"
            << "}\n";
    }

    void MarkFailed(wstring message) {
        m_failed = true;
        m_errorMessage = move(message);
    }

    void ResetWriter() {
        if (m_timingStream.is_open()) {
            m_timingStream.close();
        }

        if (m_rawStream.is_open()) {
            m_rawStream.close();
        }

        m_stagingTexture = nullptr;
        m_frameBuffer.clear();
        m_lastFrameTimestamp = -1;
    }

    bool m_enabled{};
    bool m_failed{};
    RecordingMode m_mode{ RecordingMode::RawBgra };
    filesystem::path m_baseOutputPath{};
    filesystem::path m_outputPath{};
    filesystem::path m_timingPath{};
    filesystem::path m_metadataPath{};
    wstring m_errorMessage{};
    int m_width{};
    int m_height{};
    size_t m_frameStride{};
    size_t m_frameBufferSize{};
    uint64_t m_frameCount{};
    uint64_t m_savedFrameCount{};
    int m_partIndex{};
    chrono::steady_clock::time_point m_startTime{};
    LONGLONG m_lastFrameTimestamp{ -1 };
    LONGLONG m_lastSnapshotTimestamp{ -1 };

    ofstream m_rawStream;
    ofstream m_timingStream;
    vector<BYTE> m_frameBuffer;
    winrt::com_ptr<ID3D11Texture2D> m_stagingTexture;
};

class CaptureHost {
public:
    bool Initialize(HINSTANCE instance, int showCommand, HostOptions const& options) {
        m_showCursor = options.showCursor;
        m_recorder.Configure(options.recordVideo, options.recordingMode, options.recordPath);

        WNDCLASSEXW windowClass{};
        windowClass.cbSize = sizeof(windowClass);
        windowClass.hInstance = instance;
        windowClass.lpfnWndProc = CaptureHost::WindowProc;
        windowClass.lpszClassName = kWindowClassName;
        windowClass.hCursor = LoadCursorW(nullptr, IDC_ARROW);
        windowClass.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_WINDOW + 1);

        if (!RegisterClassExW(&windowClass) && GetLastError() != ERROR_CLASS_ALREADY_EXISTS) {
            return false;
        }

        RECT bounds{ 0, 0, 1280, 720 };
        AdjustWindowRectEx(&bounds, WS_OVERLAPPEDWINDOW, FALSE, 0);

        m_window = CreateWindowExW(
            0,
            kWindowClassName,
            L"Avion Capture Preview",
            WS_OVERLAPPEDWINDOW,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            bounds.right - bounds.left,
            bounds.bottom - bounds.top,
            nullptr,
            nullptr,
            instance,
            this);

        if (!m_window) {
            return false;
        }

        ShowWindow(m_window, showCommand);
        UpdateWindow(m_window);
        return true;
    }

    bool Start() {
        if (!wgc::GraphicsCaptureSession::IsSupported()) {
            ShowError(L"Windows Graphics Capture is not supported on this system.");
            return false;
        }

        CreateDevice();
        m_captureItem = CreatePrimaryMonitorItem();
        m_captureSize = m_captureItem.Size();

        ResizeWindowForCapture(m_captureSize);
        EnsureSwapChain(m_captureSize.Width, m_captureSize.Height);
        m_recorder.Initialize(m_d3dDevice.get(), m_captureSize.Width, m_captureSize.Height);

        m_framePool = wgc::Direct3D11CaptureFramePool::CreateFreeThreaded(
            m_direct3DDevice,
            wgd::DirectXPixelFormat::B8G8R8A8UIntNormalized,
            2,
            m_captureSize);

        m_frameArrivedToken = m_framePool.FrameArrived({ this, &CaptureHost::OnFrameArrived });
        m_itemClosedToken = m_captureItem.Closed([this](auto const&, auto const&) {
            PostMessageW(m_window, WM_CLOSE, 0, 0);
        });

        m_session = m_framePool.CreateCaptureSession(m_captureItem);
        m_session.IsCursorCaptureEnabled(m_showCursor);
        m_running.store(true);
        m_session.StartCapture();

        SetWindowTextW(m_window, (L"Avion Capture Preview - running" + m_recorder.TitleSuffix()).c_str());
        return true;
    }

    void Stop() {
        scoped_lock lock(m_renderMutex);
        m_running.store(false);

        if (m_captureItem) {
            m_captureItem.Closed(m_itemClosedToken);
        }

        if (m_framePool) {
            m_framePool.FrameArrived(m_frameArrivedToken);
        }

        if (m_session) {
            m_session.Close();
            m_session = nullptr;
        }

        if (m_framePool) {
            m_framePool.Close();
            m_framePool = nullptr;
        }

        m_recorder.Stop();

        m_captureItem = nullptr;
        m_swapChain = nullptr;
        m_d3dContext = nullptr;
        m_d3dDevice = nullptr;
        m_direct3DDevice = nullptr;
    }

    int Run() {
        StartControlThread();

        MSG message{};
        while (GetMessageW(&message, nullptr, 0, 0)) {
            TranslateMessage(&message);
            DispatchMessageW(&message);
        }

        return static_cast<int>(message.wParam);
    }

private:
    void StartControlThread() {
        thread([this]() {
            HANDLE input = GetStdHandle(STD_INPUT_HANDLE);
            if (input == INVALID_HANDLE_VALUE || input == nullptr) {
                return;
            }

            string command;
            char buffer[64]{};
            DWORD bytesRead = 0;

            while (ReadFile(input, buffer, sizeof(buffer), &bytesRead, nullptr) && bytesRead > 0) {
                command.append(buffer, buffer + bytesRead);
                if (command.find("stop") != string::npos || command.find("quit") != string::npos) {
                    PostMessageW(m_window, WM_CLOSE, 0, 0);
                    return;
                }

                if (command.size() > 256) {
                    command.erase(0, command.size() - 256);
                }
            }
        }).detach();
    }

    static LRESULT CALLBACK WindowProc(HWND window, UINT message, WPARAM wParam, LPARAM lParam) {
        CaptureHost* host = nullptr;

        if (message == WM_NCCREATE) {
            auto create = reinterpret_cast<CREATESTRUCTW*>(lParam);
            host = static_cast<CaptureHost*>(create->lpCreateParams);
            SetWindowLongPtrW(window, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(host));
            host->m_window = window;
        }
        else {
            host = reinterpret_cast<CaptureHost*>(GetWindowLongPtrW(window, GWLP_USERDATA));
        }

        if (host) {
            return host->HandleMessage(message, wParam, lParam);
        }

        return DefWindowProcW(window, message, wParam, lParam);
    }

    LRESULT HandleMessage(UINT message, WPARAM wParam, LPARAM lParam) {
        switch (message) {
        case WM_ERASEBKGND:
            return 1;

        case WM_KEYDOWN:
            if (wParam == VK_ESCAPE) {
                DestroyWindow(m_window);
                return 0;
            }
            break;

        case kUpdateFpsMessage: {
            const auto fpsTenths = static_cast<int>(wParam);
            wstringstream title;
            title << L"Avion Capture Preview - " << (fpsTenths / 10) << L"." << (fpsTenths % 10) << L" FPS";
            title << m_recorder.TitleSuffix();
            SetWindowTextW(m_window, title.str().c_str());
            return 0;
        }

        case WM_DESTROY:
            Stop();
            PostQuitMessage(0);
            return 0;
        }

        return DefWindowProcW(m_window, message, wParam, lParam);
    }

    void CreateDevice() {
        constexpr D3D_FEATURE_LEVEL featureLevels[] = {
            D3D_FEATURE_LEVEL_11_1,
            D3D_FEATURE_LEVEL_11_0,
            D3D_FEATURE_LEVEL_10_1,
            D3D_FEATURE_LEVEL_10_0,
        };

        UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;

        D3D_FEATURE_LEVEL selectedFeatureLevel{};
        auto hr = D3D11CreateDevice(
            nullptr,
            D3D_DRIVER_TYPE_HARDWARE,
            nullptr,
            flags,
            featureLevels,
            ARRAYSIZE(featureLevels),
            D3D11_SDK_VERSION,
            m_d3dDevice.put(),
            &selectedFeatureLevel,
            m_d3dContext.put());

        if (FAILED(hr)) {
            winrt::check_hresult(D3D11CreateDevice(
                nullptr,
                D3D_DRIVER_TYPE_WARP,
                nullptr,
                flags,
                featureLevels,
                ARRAYSIZE(featureLevels),
                D3D11_SDK_VERSION,
                m_d3dDevice.put(),
                &selectedFeatureLevel,
                m_d3dContext.put()));
        }

        if (auto multithread = m_d3dDevice.try_as<ID3D11Multithread>()) {
            multithread->SetMultithreadProtected(TRUE);
        }

        auto dxgiDevice = m_d3dDevice.as<IDXGIDevice>();
        if (auto dxgiDevice1 = dxgiDevice.try_as<IDXGIDevice1>()) {
            dxgiDevice1->SetMaximumFrameLatency(1);
        }

        winrt::com_ptr<::IInspectable> inspectableDevice;
        winrt::check_hresult(CreateDirect3D11DeviceFromDXGIDevice(dxgiDevice.get(), inspectableDevice.put()));
        m_direct3DDevice = inspectableDevice.as<wgd11::IDirect3DDevice>();
    }

    wgc::GraphicsCaptureItem CreatePrimaryMonitorItem() {
        const POINT primaryOrigin{ 0, 0 };
        const HMONITOR monitor = MonitorFromPoint(primaryOrigin, MONITOR_DEFAULTTOPRIMARY);
        auto interop = winrt::get_activation_factory<wgc::GraphicsCaptureItem, IGraphicsCaptureItemInterop>();

        wgc::GraphicsCaptureItem item{ nullptr };
        winrt::check_hresult(interop->CreateForMonitor(
            monitor,
            winrt::guid_of<ABI::Windows::Graphics::Capture::IGraphicsCaptureItem>(),
            winrt::put_abi(item)));

        return item;
    }

    void EnsureSwapChain(int width, int height) {
        width = max(width, 1);
        height = max(height, 1);

        if (m_swapChain) {
            winrt::check_hresult(m_swapChain->ResizeBuffers(
                2,
                static_cast<UINT>(width),
                static_cast<UINT>(height),
                DXGI_FORMAT_B8G8R8A8_UNORM,
                0));
            return;
        }

        DXGI_SWAP_CHAIN_DESC1 description{};
        description.Width = static_cast<UINT>(width);
        description.Height = static_cast<UINT>(height);
        description.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        description.Stereo = FALSE;
        description.SampleDesc.Count = 1;
        description.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        description.BufferCount = 2;
        description.Scaling = DXGI_SCALING_STRETCH;
        description.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        description.AlphaMode = DXGI_ALPHA_MODE_IGNORE;

        auto dxgiDevice = m_d3dDevice.as<IDXGIDevice>();
        winrt::com_ptr<IDXGIAdapter> adapter;
        winrt::check_hresult(dxgiDevice->GetAdapter(adapter.put()));

        winrt::com_ptr<IDXGIFactory2> factory;
        winrt::check_hresult(adapter->GetParent(winrt::guid_of<IDXGIFactory2>(), factory.put_void()));

        winrt::check_hresult(factory->CreateSwapChainForHwnd(
            m_d3dDevice.get(),
            m_window,
            &description,
            nullptr,
            nullptr,
            m_swapChain.put()));
    }

    void ResizeWindowForCapture(wg::SizeInt32 size) {
        if (size.Width <= 0 || size.Height <= 0) {
            return;
        }

        constexpr int maxClientWidth = 1280;
        constexpr int maxClientHeight = 720;

        const double scale = min(
            1.0,
            min(
                static_cast<double>(maxClientWidth) / static_cast<double>(size.Width),
                static_cast<double>(maxClientHeight) / static_cast<double>(size.Height)));

        const int clientWidth = max(320, static_cast<int>(static_cast<double>(size.Width) * scale));
        const int clientHeight = max(180, static_cast<int>(static_cast<double>(size.Height) * scale));

        RECT bounds{ 0, 0, clientWidth, clientHeight };
        AdjustWindowRectEx(&bounds, WS_OVERLAPPEDWINDOW, FALSE, 0);
        SetWindowPos(
            m_window,
            nullptr,
            0,
            0,
            bounds.right - bounds.left,
            bounds.bottom - bounds.top,
            SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE);
    }

    void OnFrameArrived(wgc::Direct3D11CaptureFramePool const& sender, winrt::Windows::Foundation::IInspectable const&) {
        if (!m_running.load()) {
            return;
        }

        scoped_lock lock(m_renderMutex);
        auto frame = sender.TryGetNextFrame();
        if (!frame) {
            return;
        }

        const auto contentSize = frame.ContentSize();
        if (contentSize.Width != m_captureSize.Width || contentSize.Height != m_captureSize.Height) {
            m_captureSize = contentSize;
            EnsureSwapChain(m_captureSize.Width, m_captureSize.Height);
            ResizeWindowForCapture(m_captureSize);
            m_framePool.Recreate(
                m_direct3DDevice,
                wgd::DirectXPixelFormat::B8G8R8A8UIntNormalized,
                2,
                m_captureSize);
        }

        auto sourceTexture = GetDXGIInterfaceFromObject<ID3D11Texture2D>(frame.Surface());
        winrt::com_ptr<ID3D11Texture2D> backBuffer;
        winrt::check_hresult(m_swapChain->GetBuffer(0, winrt::guid_of<ID3D11Texture2D>(), backBuffer.put_void()));

        m_recorder.WriteFrame(m_d3dContext.get(), sourceTexture.get());
        m_d3dContext->CopyResource(backBuffer.get(), sourceTexture.get());
        winrt::check_hresult(m_swapChain->Present(1, 0));

        TrackFps();
    }

    void TrackFps() {
        using clock = chrono::steady_clock;
        const auto now = clock::now();

        if (m_lastFpsTime.time_since_epoch().count() == 0) {
            m_lastFpsTime = now;
        }

        ++m_framesSinceLastFps;
        const auto elapsed = chrono::duration_cast<chrono::milliseconds>(now - m_lastFpsTime);
        if (elapsed.count() < 1000) {
            return;
        }

        const double fps = static_cast<double>(m_framesSinceLastFps) * 1000.0 / static_cast<double>(elapsed.count());
        m_framesSinceLastFps = 0;
        m_lastFpsTime = now;
        PostMessageW(m_window, kUpdateFpsMessage, static_cast<WPARAM>(fps * 10.0), 0);
    }

    HWND m_window{};
    bool m_showCursor{};
    atomic_bool m_running{ false };
    mutex m_renderMutex;

    winrt::com_ptr<ID3D11Device> m_d3dDevice;
    winrt::com_ptr<ID3D11DeviceContext> m_d3dContext;
    winrt::com_ptr<IDXGISwapChain1> m_swapChain;

    wgd11::IDirect3DDevice m_direct3DDevice{ nullptr };
    wgc::GraphicsCaptureItem m_captureItem{ nullptr };
    wgc::Direct3D11CaptureFramePool m_framePool{ nullptr };
    wgc::GraphicsCaptureSession m_session{ nullptr };
    winrt::event_token m_frameArrivedToken{};
    winrt::event_token m_itemClosedToken{};
    wg::SizeInt32 m_captureSize{};

    chrono::steady_clock::time_point m_lastFpsTime{};
    int m_framesSinceLastFps{};
    LosslessFrameRecorder m_recorder{};
};
}

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR, int showCommand) {
    winrt::init_apartment(winrt::apartment_type::multi_threaded);

    try {
        const HostOptions options = ParseHostOptions();

        CaptureHost host;
        if (!host.Initialize(instance, showCommand, options)) {
            ShowError(L"Failed to create the capture preview window.");
            return 1;
        }

        if (!host.Start()) {
            return 2;
        }

        return host.Run();
    }
    catch (winrt::hresult_error const& error) {
        ShowError(L"Windows capture failed: " + HResultMessage(error));
        return 3;
    }
    catch (exception const& error) {
        wstring message = L"Native capture host failed: ";
        message += winrt::to_hstring(error.what()).c_str();
        ShowError(message);
        return 4;
    }
}
