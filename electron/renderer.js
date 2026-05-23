const toggleButton = document.getElementById('capture-toggle')
const toggleLabel = document.getElementById('capture-toggle-label')
const statusPill = document.getElementById('status-pill')
const captureMessage = document.getElementById('capture-message')
const oneFpsToggle = document.getElementById('one-fps-toggle')

let isCaptureEnabled = false
let isBusy = false

const statusCopy = {
  running: 'Running',
  stopping: 'Stopping',
  stopped: 'Idle',
  exited: 'Exited',
  failed: 'Failed',
  'missing-host': 'Build required'
}

function recordingPathMessage (path) {
  return path ? ` Saving to: ${path}` : ''
}

function renderToggleState (state = {}) {
  const status = state.status ?? (isCaptureEnabled ? 'running' : 'stopped')

  isCaptureEnabled = Boolean(state.enabled)
  toggleLabel.textContent = isCaptureEnabled ? 'ON' : 'OFF'
  toggleButton.classList.toggle('is-on', isCaptureEnabled)
  toggleButton.setAttribute('aria-pressed', String(isCaptureEnabled))
  toggleButton.disabled = isBusy
  oneFpsToggle.disabled = isBusy || isCaptureEnabled

  statusPill.textContent = statusCopy[status] ?? status
  statusPill.dataset.status = status

  if (state.message) {
    captureMessage.textContent = state.message
  } else if (status === 'running') {
    captureMessage.textContent = `Native preview window active${state.pid ? ` (PID ${state.pid})` : ''}.${recordingPathMessage(state.recordingPath)}`
  } else if (status === 'stopping') {
    captureMessage.textContent = `Finalizing capture${state.recordingPath ? `: ${state.recordingPath}` : ''}`
  } else if (status === 'missing-host') {
    captureMessage.textContent = 'Run npm run build:native before starting capture.'
  } else if (status === 'exited') {
    captureMessage.textContent = `Native capture host exited${state.exitCode === null ? '' : ` with code ${state.exitCode}`}`
  } else if (status === 'stopped' && state.recordingPath) {
    captureMessage.textContent = `Saved capture: ${state.recordingPath}`
  } else {
    captureMessage.textContent = 'Ready'
  }
}

toggleButton.addEventListener('click', async () => {
  if (!window.captureControl || isBusy) {
    return
  }

  isBusy = true
  renderToggleState({ enabled: isCaptureEnabled, status: isCaptureEnabled ? 'running' : 'stopped' })

  try {
    const state = await window.captureControl.setEnabled(!isCaptureEnabled, {
      oneFramePerSecond: oneFpsToggle.checked
    })
    renderToggleState(state)
  } catch (error) {
    renderToggleState({
      enabled: false,
      status: 'failed',
      message: error.message
    })
  } finally {
    isBusy = false
    toggleButton.disabled = false
    oneFpsToggle.disabled = isCaptureEnabled
  }
})

async function initialize () {
  if (!window.captureControl) {
    renderToggleState({
      enabled: false,
      status: 'failed',
      message: 'Capture bridge is unavailable.'
    })
    return
  }

  window.captureControl.onStatus(renderToggleState)
  renderToggleState(await window.captureControl.getStatus())
}

initialize()
