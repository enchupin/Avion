const { app, BrowserWindow, ipcMain } = require('electron')
const { spawn } = require('child_process')
const fs = require('fs')
const path = require('path')

let captureEnabled = false
let captureProcess = null
let lastRecordingPath = null
let lastRecordingMode = null
let captureSessionCounter = 0
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
    recordingPath: lastRecordingPath,
    recordingMode: lastRecordingMode,
    ...overrides
  }
}

function timestampForFileName (date = new Date()) {
  const pad = (value) => String(value).padStart(2, '0')
  return [
    date.getFullYear(),
    pad(date.getMonth() + 1),
    pad(date.getDate())
  ].join('') + '-' + [
    pad(date.getHours()),
    pad(date.getMinutes()),
    pad(date.getSeconds())
  ].join('')
}

function createRecordingPath (recordingMode) {
  const recordingDir = path.join(app.getPath('videos'), 'Avion Captures')
  fs.mkdirSync(recordingDir, { recursive: true })
  const timestamp = timestampForFileName()
  const sessionId = String(++captureSessionCounter).padStart(3, '0')
  const baseName = `avion-capture-${timestamp}-${sessionId}`

  for (let attempt = 0; attempt < 1000; ++attempt) {
    const suffix = attempt === 0 ? '' : `-${String(attempt + 1).padStart(3, '0')}`
    const uniqueBaseName = `${baseName}${suffix}`
    const candidate = recordingMode === 'png-1fps'
      ? path.join(recordingDir, `${uniqueBaseName}-1fps`)
      : path.join(recordingDir, `${uniqueBaseName}.bgra`)

    if (!fs.existsSync(candidate)) {
      return candidate
    }
  }

  const fallbackName = `avion-capture-${timestamp}-${sessionId}-${Date.now()}`
  return recordingMode === 'png-1fps'
    ? path.join(recordingDir, `${fallbackName}-1fps`)
    : path.join(recordingDir, `${fallbackName}.bgra`)
}

function startCaptureHost (options = {}) {
  if (captureProcess && !captureProcess.killed) {
    return captureState({ status: captureEnabled ? 'running' : 'stopping' })
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
    const recordingMode = options.oneFramePerSecond ? 'png-1fps' : 'raw-bgra'
    lastRecordingMode = recordingMode
    lastRecordingPath = createRecordingPath(recordingMode)
    captureProcess = spawn(hostPath, ['--record', '--record-mode', recordingMode, '--record-path', lastRecordingPath], {
      cwd: path.dirname(hostPath),
      detached: false,
      stdio: ['pipe', 'ignore', 'ignore'],
      windowsHide: false
    })
  } catch (error) {
    captureEnabled = false
    captureProcess = null
    lastRecordingPath = null
    lastRecordingMode = null
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
  return captureState({ status: 'running', recordingMode: options.oneFramePerSecond ? 'png-1fps' : 'raw-bgra' })
}

function stopCaptureHost () {
  if (captureProcess && !captureProcess.killed) {
    const child = captureProcess
    captureEnabled = false

    try {
      child.stdin?.write('stop\n')
      child.stdin?.end()
    } catch {
      child.kill()
    }

    setTimeout(() => {
      if (captureProcess === child && !child.killed) {
        child.kill()
      }
    }, 3000)

    return captureState({ status: 'stopping' })
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

ipcMain.handle('capture:set-enabled', (_event, enabled, options = {}) => {
  const state = enabled ? startCaptureHost(options) : stopCaptureHost()
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
