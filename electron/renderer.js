const toggleButton = document.getElementById('capture-toggle')
const toggleLabel = document.getElementById('capture-toggle-label')
const statusPill = document.getElementById('status-pill')
const captureMessage = document.getElementById('capture-message')
const oneFpsToggle = document.getElementById('one-fps-toggle')
const videoExtractButton = document.getElementById('video-extract-button')
const videoOpenOutputButton = document.getElementById('video-open-output-button')
const videoStatusPill = document.getElementById('video-status-pill')
const videoFpsValue = document.getElementById('video-fps-value')
const videoFrameCount = document.getElementById('video-frame-count')
const videoDuration = document.getElementById('video-duration')
const videoSize = document.getElementById('video-size')
const videoProgress = document.getElementById('video-progress')
const videoMessage = document.getElementById('video-message')

let isCaptureEnabled = false
let isBusy = false
let isVideoBusy = false
let lastVideoOutputDir = null

const statusCopy = {
  running: 'Running',
  stopping: 'Stopping',
  stopped: 'Idle',
  exited: 'Exited',
  failed: 'Failed',
  'missing-host': 'Build required'
}

const videoStatusCopy = {
  idle: 'Idle',
  analyzing: 'Analyzing',
  extracting: 'Extracting',
  completed: 'Complete',
  failed: 'Failed',
  cancelled: 'Idle'
}

function recordingPathMessage (path) {
  return path ? ` Saving to: ${path}` : ''
}

function formatNumber (value) {
  return Number.isFinite(value) ? value.toLocaleString() : '--'
}

function formatFps (value) {
  return Number.isFinite(value) ? value.toFixed(3).replace(/0+$/, '').replace(/\.$/, '') : '--'
}

function formatDuration (seconds) {
  if (!Number.isFinite(seconds) || seconds <= 0) {
    return '--'
  }

  const rounded = Math.round(seconds)
  const hours = Math.floor(rounded / 3600)
  const minutes = Math.floor((rounded % 3600) / 60)
  const remainingSeconds = rounded % 60

  if (hours > 0) {
    return `${hours}:${String(minutes).padStart(2, '0')}:${String(remainingSeconds).padStart(2, '0')}`
  }

  return `${minutes}:${String(remainingSeconds).padStart(2, '0')}`
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

function renderVideoFramesState (state = {}) {
  const status = state.status ?? 'idle'
  const metadata = state.metadata ?? null

  if (state.outputDir) {
    lastVideoOutputDir = state.outputDir
  }

  videoStatusPill.textContent = videoStatusCopy[status] ?? status
  videoStatusPill.dataset.status = status
  videoExtractButton.disabled = isVideoBusy
  videoOpenOutputButton.hidden = !lastVideoOutputDir

  if (metadata) {
    videoFpsValue.textContent = metadata.variableFrameRate ? `Variable ${formatFps(metadata.fps)} avg` : formatFps(metadata.fps)
    videoDuration.textContent = formatDuration(metadata.durationSeconds)
    videoSize.textContent = metadata.width && metadata.height ? `${metadata.width}x${metadata.height}` : '--'
    videoFrameCount.textContent = formatNumber(state.frameCount ?? metadata.estimatedFrameCount)
  } else if (status === 'idle' || status === 'cancelled') {
    videoFpsValue.textContent = '--'
    videoFrameCount.textContent = '--'
    videoDuration.textContent = '--'
    videoSize.textContent = '--'
  }

  if (Number.isFinite(state.percent)) {
    videoProgress.hidden = false
    videoProgress.value = state.percent
  } else if (status === 'idle' || status === 'cancelled') {
    videoProgress.hidden = true
    videoProgress.value = 0
  }

  if (state.message) {
    videoMessage.textContent = state.message
  } else if (status === 'analyzing') {
    videoMessage.textContent = `Analyzing source: ${state.videoPath}`
  } else if (status === 'extracting') {
    const frameText = Number.isFinite(state.frame) ? ` Frame ${formatNumber(state.frame)}.` : ''
    const timingText = metadata?.variableFrameRate ? ' Timing CSV will be preserved.' : ''
    videoMessage.textContent = `Writing PNG frames to: ${state.outputDir}.${frameText}${timingText}`
  } else if (status === 'completed') {
    videoProgress.hidden = false
    videoProgress.value = 100
    videoFrameCount.textContent = formatNumber(state.frameCount)
    const timingText = metadata?.variableFrameRate ? ' Timing saved to frames.csv.' : ''
    videoMessage.textContent = `Saved ${formatNumber(state.frameCount)} PNG frames to: ${state.outputDir}.${timingText}`
  } else if (status === 'cancelled') {
    videoMessage.textContent = 'Ready'
  } else if (status === 'failed') {
    videoMessage.textContent = 'Frame extraction failed.'
  } else {
    videoMessage.textContent = 'Ready'
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

videoExtractButton.addEventListener('click', async () => {
  if (!window.videoFrames || isVideoBusy) {
    return
  }

  isVideoBusy = true
  lastVideoOutputDir = null
  renderVideoFramesState({ status: 'idle' })

  try {
    const state = await window.videoFrames.selectAndExtract()
    renderVideoFramesState(state)
  } catch (error) {
    renderVideoFramesState({
      status: 'failed',
      message: error.message
    })
  } finally {
    isVideoBusy = false
    videoExtractButton.disabled = false
    videoOpenOutputButton.hidden = !lastVideoOutputDir
  }
})

videoOpenOutputButton.addEventListener('click', async () => {
  if (!window.videoFrames || !lastVideoOutputDir) {
    return
  }

  const result = await window.videoFrames.openOutput(lastVideoOutputDir)
  if (!result.success && result.message) {
    renderVideoFramesState({
      status: 'failed',
      message: result.message
    })
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

  if (!window.videoFrames) {
    renderVideoFramesState({
      status: 'failed',
      message: 'Video bridge is unavailable.'
    })
    return
  }

  window.videoFrames.onStatus((state) => {
    if (state.status === 'analyzing' || state.status === 'extracting') {
      isVideoBusy = true
    } else if (state.status === 'completed' || state.status === 'failed' || state.status === 'cancelled') {
      isVideoBusy = false
    }
    renderVideoFramesState(state)
  })
  renderVideoFramesState()
}

initialize()
