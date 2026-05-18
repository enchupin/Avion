const { app, BrowserWindow, ipcMain } = require('electron')
const path = require('path')

let captureEnabled = false

function createWindow () {
  const win = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js')
    }
  })
  win.loadFile('index.html')
}

ipcMain.handle('capture:set-enabled', (_event, enabled) => {
  captureEnabled = Boolean(enabled)

  // Connect the C++ capture start/stop call here later.
  console.log(`Capture ${captureEnabled ? 'enabled' : 'disabled'}`)

  return { enabled: captureEnabled }
})

app.whenReady().then(() => {
  createWindow()
})
app.on('window-all-closed', function () {
  if (process.platform !== 'darwin') app.quit()
})
