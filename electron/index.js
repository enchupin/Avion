const { app, BrowserWindow, ipcMain } = require('electron')
const { spawn } = require('child_process')
const fs = require('fs')
const path = require('path')

let captureEnabled = false
let captureProcess = null
const windows = new Set()

function getCaptureHostCandidates () {
  const executable = 'AvionCaptureHost.exe'
  const unpackedPath = process.resourcesPath
    ? path.join(process.resourcesPath, 'native', executable)
    : null

  return [
    path.join(__dirname, 'native', 'capture-host', 'build', 'Release', executable),
    path.join(__dirname, 'native', 'capture-host', 'build', 'Debug', executable),
    unpackedPath
  ].filter(Boolean)
}

function findCaptureHost () {
  return getCaptureHostCandidates().find((candidate) => fs.existsSync(candidate))
}

function broadcastCaptureStatus (status) {
  for (const win of windows) {
    if (!win.isDestroyed()) {
      win.webContents.send('capture:status', status)
    }
  }
}

function captureState (overrides = {}) {
  return {
    enabled: captureEnabled,
    pid: captureProcess?.pid ?? null,
    ...overrides
  }
}

function startCaptureHost () {
  if (captureProcess && !captureProcess.killed) {
    return captureState({ status: 'running' })
  }

  const hostPath = findCaptureHost()
  if (!hostPath) {
    captureEnabled = false
    return captureState({
      status: 'missing-host',
      message: 'Native capture host is not built.'
    })
  }

  try {
    captureProcess = spawn(hostPath, [], {
      cwd: path.dirname(hostPath),
      detached: false,
      stdio: 'ignore',
      windowsHide: false
    })
  } catch (error) {
    captureEnabled = false
    captureProcess = null
    return captureState({
      status: 'failed',
      message: error.message
    })
  }

  captureEnabled = true
  const child = captureProcess

  child.once('error', (error) => {
    if (captureProcess !== child) {
      return
    }

    captureEnabled = false
    captureProcess = null
    broadcastCaptureStatus(captureState({
      status: 'failed',
      message: error.message
    }))
  })

  child.once('exit', (code, signal) => {
    if (captureProcess !== child) {
      return
    }

    captureEnabled = false
    captureProcess = null
    broadcastCaptureStatus(captureState({
      status: code === 0 || signal === 'SIGTERM' ? 'stopped' : 'exited',
      exitCode: code,
      signal
    }))
  })

  child.unref()
  return captureState({ status: 'running' })
}

function stopCaptureHost () {
  if (captureProcess && !captureProcess.killed) {
    captureProcess.kill()
  }

  captureProcess = null
  captureEnabled = false
  return captureState({ status: 'stopped' })
}

function createWindow () {
  const win = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js')
    }
  })
  windows.add(win)
  win.on('closed', () => windows.delete(win))
  win.loadFile('index.html')
}

ipcMain.handle('capture:set-enabled', (_event, enabled) => {
  const state = enabled ? startCaptureHost() : stopCaptureHost()
  broadcastCaptureStatus(state)
  return state
})

ipcMain.handle('capture:get-status', () => {
  return captureState({ status: captureEnabled ? 'running' : 'stopped' })
})

app.whenReady().then(() => {
  createWindow()
})

app.on('before-quit', () => {
  stopCaptureHost()
})

app.on('window-all-closed', function () {
  if (process.platform !== 'darwin') app.quit()
})
