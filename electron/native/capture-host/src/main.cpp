#include <windows.h>
#include <windows.graphics.capture.interop.h>
#include <windows.graphics.directx.direct3d11.interop.h>

#include <d3d11_4.h>
#include <dxgi1_6.h>

#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Graphics.Capture.h>
#include <winrt/Windows.Graphics.DirectX.Direct3D11.h>
#include <winrt/Windows.Graphics.DirectX.h>
#include <winrt/Windows.Graphics.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <exception>
#include <mutex>
#include <sstream>
#include <string>

using namespace std;

namespace wgc = winrt::Windows::Graphics::Capture;
namespace wg = winrt::Windows::Graphics;
namespace wgd = winrt::Windows::Graphics::DirectX;
namespace wgd11 = winrt::Windows::Graphics::DirectX::Direct3D11;

namespace {
constexpr wchar_t kWindowClassName[] = L"AvionCaptureHostWindow";
constexpr UINT kUpdateFpsMessage = WM_APP + 1;

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

bool HasArgument(PWSTR commandLine, wstring const& name) {
    if (commandLine == nullptr) {
        return false;
    }

    return wstring(commandLine).find(name) != wstring::npos;
}

class CaptureHost {
public:
    bool Initialize(HINSTANCE instance, int showCommand, bool showCursor) {
        m_showCursor = showCursor;

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

        SetWindowTextW(m_window, L"Avion Capture Preview - running");
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

        m_captureItem = nullptr;
        m_swapChain = nullptr;
        m_d3dContext = nullptr;
        m_d3dDevice = nullptr;
        m_direct3DDevice = nullptr;
    }

    int Run() {
        MSG message{};
        while (GetMessageW(&message, nullptr, 0, 0)) {
            TranslateMessage(&message);
            DispatchMessageW(&message);
        }

        return static_cast<int>(message.wParam);
    }

private:
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
};
}

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR commandLine, int showCommand) {
    winrt::init_apartment(winrt::apartment_type::multi_threaded);

    try {
        CaptureHost host;
        if (!host.Initialize(instance, showCommand, HasArgument(commandLine, L"--show-cursor"))) {
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
