const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('captureControl', {
  setEnabled: (enabled) => ipcRenderer.invoke('capture:set-enabled', enabled)
})
