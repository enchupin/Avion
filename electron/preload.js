const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('captureControl', {
  setEnabled: (enabled, options) => ipcRenderer.invoke('capture:set-enabled', enabled, options),
  getStatus: () => ipcRenderer.invoke('capture:get-status'),
  onStatus: (callback) => {
    const handler = (_event, status) => callback(status)
    ipcRenderer.on('capture:status', handler)
    return () => ipcRenderer.removeListener('capture:status', handler)
  }
})

contextBridge.exposeInMainWorld('videoFrames', {
  selectAndExtract: () => ipcRenderer.invoke('video-frames:select-and-extract'),
  openOutput: (outputDir) => ipcRenderer.invoke('video-frames:open-output', outputDir),
  onStatus: (callback) => {
    const handler = (_event, status) => callback(status)
    ipcRenderer.on('video-frames:status', handler)
    return () => ipcRenderer.removeListener('video-frames:status', handler)
  }
})
