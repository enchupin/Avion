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

Press the ON/OFF button in Electron to start or stop the native capture preview window. The first native host captures the primary display through Windows Graphics Capture and renders frames through Direct3D 11.
