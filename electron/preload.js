const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('performanceEvaluation', {
  getState: () => ipcRenderer.invoke('performance:get-state'),
  select: (type) => ipcRenderer.invoke('performance:select', type),
  start: (options) => ipcRenderer.invoke('performance:start', options),
  cancel: () => ipcRenderer.invoke('performance:cancel'),
  record: () => ipcRenderer.invoke('performance:record'),
  stopRecording: () => ipcRenderer.invoke('performance:record-stop'),
  open: (kind) => ipcRenderer.invoke('performance:open', kind),
  onStatus: (callback) => {
    const handler = (_event, status) => callback(status)
    ipcRenderer.on('performance:status', handler)
    return () => ipcRenderer.removeListener('performance:status', handler)
  }
})

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
