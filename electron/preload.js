const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('srEnvironment', {
  getState: () => ipcRenderer.invoke('sr-environment:get-state'),
  check: () => ipcRenderer.invoke('sr-environment:check'),
  install: component => ipcRenderer.invoke('sr-environment:install', component),
  cancel: () => ipcRenderer.invoke('sr-environment:cancel'),
  onStatus: callback => {
    const handler = (_event, status) => callback(status)
    ipcRenderer.on('sr-environment:status', handler)
    return () => ipcRenderer.removeListener('sr-environment:status', handler)
  }
})

contextBridge.exposeInMainWorld('doomGame', {
  getState: () => ipcRenderer.invoke('doom:get-state'),
  play: options => ipcRenderer.invoke('doom:play', options),
  stopGame: () => ipcRenderer.invoke('doom:stop'),
  onStatus: (callback) => {
    const handler = (_event, status) => callback(status)
    ipcRenderer.on('doom:status', handler)
    return () => ipcRenderer.removeListener('doom:status', handler)
  }
})
