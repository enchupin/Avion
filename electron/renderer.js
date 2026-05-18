const toggleButton = document.getElementById('capture-toggle')

let isCaptureEnabled = false

function renderToggleState () {
  toggleButton.textContent = isCaptureEnabled ? 'ON' : 'OFF'
  toggleButton.classList.toggle('is-on', isCaptureEnabled)
  toggleButton.setAttribute('aria-pressed', String(isCaptureEnabled))
}

toggleButton.addEventListener('click', async () => {
  isCaptureEnabled = !isCaptureEnabled
  renderToggleState()

  if (window.captureControl) {
    await window.captureControl.setEnabled(isCaptureEnabled)
  }
})

renderToggleState()
