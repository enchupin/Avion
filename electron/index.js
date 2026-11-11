const { app, BrowserWindow, ipcMain } = require('electron')
const path = require('path')
const { registerDoomGame } = require('./doom-game')
const { registerEnvironment } = require('./sr-environment')

const windows = new Set()

function createWindow () {
  const win = new BrowserWindow({
    width: 1280,
    height: 920,
    show: process.env.AVION_TEST_HEADLESS !== '1',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js')
    }
  })
  windows.add(win)
  win.on('closed', () => windows.delete(win))
  win.loadFile(path.join(__dirname, 'index.html'))
}

app.whenReady().then(() => {
  let doomGame
  const environment = registerEnvironment({ app, ipcMain, windows,
    isInUse: () => doomGame?.isRunning() })
  doomGame = registerDoomGame({ app, ipcMain, windows, environment })
  createWindow()
})

app.on('window-all-closed', function () {
  if (process.platform !== 'darwin') app.quit()
})
