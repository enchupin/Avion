const environmentBridge = window.srEnvironment
const srElement = name => document.getElementById(`sr-${name}`)
let srState = { status: 'idle' }
let srGameBusy = false
const srComponents = [['python', 'Python 실행 환경'], ['numpy', 'NumPy'], ['cv2', 'OpenCV'], ['skimage', 'scikit-image'], ['torch', 'PyTorch']]

function showEnvironment (state) {
  if (state.error) { srElement('message').textContent = state.error; return }
  srState = state
  const busy = ['checking', 'installing'].includes(state.status)
  const checked = !['idle', 'checking'].includes(state.status)
  srElement('check').disabled = busy || srGameBusy
  srElement('check').textContent = busy ? '환경 작업 중…' : '설치 여부 확인'
  srElement('cancel').hidden = !busy
  srElement('progress').hidden = !busy
  if (state.progress == null) srElement('progress').removeAttribute('value')
  else { srElement('progress').max = 100; srElement('progress').value = state.progress }
  srElement('message').textContent = state.message || '설치 여부 확인을 눌러 시작하세요.'
  srElement('components').replaceChildren()
  for (const [key, label] of srComponents) {
    const version = key === 'python' ? state.report?.version : state.report?.packages?.[key]
    const item = document.createElement('li')
    const name = document.createElement('strong')
    name.textContent = label
    const status = document.createElement('span')
    status.textContent = version ? `설치됨 · ${version}` : checked ? '미설치 또는 실행 불가' : '확인 전'
    status.className = version ? 'sr-installed' : 'sr-missing'
    item.append(name, status)
    if (checked && !version) {
      const install = document.createElement('button')
      install.type = 'button'
      install.className = 'action-button secondary'
      install.textContent = `${label} 설치`
      install.disabled = busy || srGameBusy || (key !== 'python' && !state.report?.version)
      install.addEventListener('click', () => environmentAction(() => environmentBridge.install(key)))
      item.append(install)
    }
    srElement('components').append(item)
  }
  srElement('gpu').textContent = `GPU SR: ${state.report?.gpuMessage || (checked ? 'Python과 PyTorch 설치 후 확인할 수 있습니다.' : '아직 확인하지 않았습니다.')}`
  srElement('gpu').dataset.available = state.report?.cuda ? 'true' : 'false'
  srElement('path').textContent = `사용 Python: ${state.report?.python || '없음'}\n자동 설치 위치: ${state.installPath || '확인 전'}`
  srElement('details').textContent = (state.report?.errors || []).join('\n')
  srElement('log').textContent = state.log || ''
  updateDoomControls(state)
}

async function environmentAction (action) {
  try { showEnvironment(await action()) } catch (error) { srElement('message').textContent = error.message }
}
srElement('check').addEventListener('click', () => environmentAction(() => environmentBridge.check()))
srElement('cancel').addEventListener('click', () => environmentAction(() => environmentBridge.cancel()))
if (environmentBridge) {
  environmentBridge.onStatus(showEnvironment)
  environmentBridge.getState().then(showEnvironment).catch(error => { srElement('message').textContent = error.message })
  const updateGame = state => {
    srGameBusy = ['playing', 'playing-stop'].includes(state.status)
    showEnvironment(srState)
  }
  window.doomGame.onStatus(updateGame)
  window.doomGame.getState().then(updateGame)
} else {
  srElement('check').disabled = true
  srElement('message').textContent = 'Avion 앱에서 실행 환경을 확인하세요.'
}
