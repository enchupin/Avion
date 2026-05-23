# Avion

Electron control surface with a native C++ Windows Graphics Capture preview host.

## Requirements

- Windows 10 1903 or newer
- Visual Studio 2022 Build Tools
- Desktop development with C++ workload
- Windows 10/11 SDK

## Run

```powershell
cd electron
npm install
npm run build:native
npm start
```

Press the ON/OFF button in Electron to start or stop the native capture preview window. The native host captures the primary display through Windows Graphics Capture and renders frames through Direct3D 11.

By default, each capture writes every captured frame as lossless raw BGRA data:

- `avion-capture-YYYYMMDD-HHMMSS-001.bgra`: uncompressed BGRA frame data
- `avion-capture-YYYYMMDD-HHMMSS-001.csv`: per-frame timestamp and instantaneous FPS
- `avion-capture-YYYYMMDD-HHMMSS-001.json`: width, height, pixel format, frame count, and duration metadata

Uncompressed capture files are large. A 1920x1080 capture at 60 FPS writes about 475 MiB per second before filesystem overhead.

Enable `1 FPS PNG` before starting capture to save one lossless PNG per second instead. That mode creates a new folder for each ON/OFF capture session, such as `avion-capture-YYYYMMDD-HHMMSS-001-1fps`, with PNG frames, `frames.csv`, and `capture.json`.

## Extract Recorded Frames

Install `ffmpeg` and run:

```powershell
cd electron
npm run extract:frames -- -CapturePath "C:\Users\you\Videos\Avion Captures\avion-capture-20260522-120000.bgra"
```

The script uses one output image per captured frame. Because the `.bgra` file is uncompressed, the extracted PNG frames preserve the captured pixel values without video compression loss.

You can also double-click `electron\extract-frames.bat`. It automatically uses the newest raw capture or 1 FPS PNG folder in `Videos\Avion Captures`.
