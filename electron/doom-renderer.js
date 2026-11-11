const doomBridge = window.doomGame
let doomBusy = false
let doomEnvironment = { ready: false }
const doomElement = id => document.getElementById(`doom-${id}`)
const doomDescriptions = {
  'imdn-roi': '몬스터·아이템 영역에 IMDN FP32, 배경에 바이큐빅을 적용합니다. ROI가 넓으면 전체 SR로 전환합니다.',
  'imdn-roi-fp16': '몬스터·아이템 영역에 IMDN FP16, 배경에 바이큐빅을 적용합니다. CUDA GPU가 필요하며 ROI가 넓으면 전체 SR로 전환합니다.',
  imdn: '전체 게임 프레임에 IMDN FP32를 적용합니다.',
  'imdn-fp16': '전체 게임 프레임에 IMDN FP16을 적용합니다. CUDA GPU가 필요합니다.',
  bicubic: '전체 게임 프레임을 CPU 바이큐빅 보간으로 확대합니다. IMDN 모델을 사용하지 않습니다.',
  bilinear: '전체 게임 프레임을 CPU 바이리니어 보간으로 확대합니다. IMDN 모델을 사용하지 않습니다.',
  nearest: '가장 가까운 픽셀을 복제해 확대합니다. 픽셀 경계가 선명하게 유지됩니다.'
}

function updateDoomControls (environment = doomEnvironment) {
  doomEnvironment = environment
  const busy = ['checking', 'installing'].includes(environment.status)
  const cuda = Boolean(environment.report?.cuda)
  const method = doomElement('method')
  const device = doomElement('device')
  const halfAvailable = cuda && device.value !== 'cpu'
  for (const option of method.options) option.disabled = option.value.endsWith('fp16') && !halfAvailable
  device.querySelector('option[value="cuda"]').disabled = !cuda
  doomElement('settings').disabled = doomBusy || busy
  device.disabled = !method.value.startsWith('imdn')
  doomElement('debug').disabled = !method.value.includes('roi')
  const invalidDevice = method.value.startsWith('imdn') && device.value === 'cuda' && !cuda
  const invalidHalf = method.value.endsWith('fp16') && !halfAvailable
  // Keep a user's selection visible if its GPU becomes unavailable; never switch methods silently.
  doomElement('play').disabled = doomBusy || busy || !environment.ready || invalidDevice || invalidHalf
  doomElement('description').textContent = doomDescriptions[method.value]
  doomElement('availability').textContent = busy ? 'SR 실행 환경을 확인·설치하고 있습니다.'
    : !environment.ready ? 'SR 실행 환경에서 설치 여부를 확인하고 미설치 항목을 설치하세요.'
      : invalidHalf ? 'FP16에는 CUDA GPU가 필요합니다. 실행 장치를 자동 선택 또는 CUDA GPU로 변경하세요.'
        : invalidDevice ? '현재 CUDA GPU를 사용할 수 없습니다. CPU 또는 자동 선택을 선택하세요.'
          : !cuda ? 'CPU 실행 가능 · FP16은 CUDA GPU 확인 후 선택할 수 있습니다.'
            : '실행 준비 완료 · FP16은 자동 선택 / CUDA GPU에서 사용할 수 있습니다.'
}

function showDoomError (error) {
  doomElement('message').textContent = error.message || error.error || String(error)
}

function renderDoomState (state) {
  if (state.error) { showDoomError(state); return }
  doomBusy = ['playing', 'playing-stop'].includes(state.status)
  doomElement('play').textContent = doomBusy ? 'DOOM 실행 중' : '선택한 방식으로 DOOM 실행'
  doomElement('stop').hidden = !doomBusy
  doomElement('stop').disabled = state.status === 'playing-stop'
  doomElement('message').textContent = state.message || ''
  const labels = { idle: '대기', playing: 'DOOM 실행 중', 'playing-stop': 'DOOM 종료 중', failed: '실패' }
  doomElement('status').textContent = labels[state.status] || '대기'
  doomElement('status').dataset.status = state.status
  updateDoomControls()
}

doomElement('play').addEventListener('click', async () => {
  if (doomElement('play').disabled) return
  doomElement('play').disabled = true
  try {
    renderDoomState(await doomBridge.play({ method: doomElement('method').value,
      scale: Number(doomElement('scale').value), device: doomElement('device').value,
      debug: doomElement('debug').checked }))
  } catch (error) { showDoomError(error) } finally { updateDoomControls() }
})
doomElement('stop').addEventListener('click', async () => {
  try { renderDoomState(await doomBridge.stopGame()) } catch (error) { showDoomError(error) }
})
for (const id of ['method', 'device']) doomElement(id).addEventListener('change', () => updateDoomControls())

if (doomBridge) {
  doomBridge.onStatus(renderDoomState)
  doomBridge.getState().then(renderDoomState).catch(showDoomError)
} else {
  doomElement('play').disabled = true
  showDoomError(new Error('DOOM 실행 연결을 사용할 수 없습니다. Avion 앱에서 실행하세요.'))
}
