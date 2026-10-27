const performanceBridge = window.performanceEvaluation
const benchmarkElement = id => document.getElementById(`benchmark-${id}`)
const benchmarkNumber = (value, digits = 2) => Number.isFinite(value) ? value.toLocaleString('ko-KR', { maximumFractionDigits: digits, minimumFractionDigits: digits }) : 'N/A'
let benchmarkResultId = null
let benchmarkBusy = false
let benchmarkSource = null
let benchmarkRoiAvailable = false

function showBenchmarkError (error) {
  benchmarkElement('message').textContent = error.message || error.error || String(error)
}

function sourceDescription (state) {
  if (state.source) benchmarkElement('source').textContent = `입력: ${state.source} · 가중치: ${state.checkpoint || '기본 IMDN (배율에 맞춰 자동 선택)'}`
  if (typeof state.roiAvailable === 'boolean') {
    benchmarkRoiAvailable = state.roiAvailable
    if (benchmarkSource !== state.source) benchmarkElement('roi').checked = benchmarkRoiAvailable
    benchmarkSource = state.source
    benchmarkElement('roi').disabled = !benchmarkRoiAvailable
  }
}

function renderBenchmarkResults (result, previews) {
  const context = result.dataset
  const sizes = [...new Set(context.images.map(image => `${image.input_size.join('×')} → ${image.output_size.join('×')}`))]
  benchmarkElement('context').textContent = `${context.label} · ${context.frames}프레임 · ${sizes.join(', ')} · ${result.environment.device_name} · ${new Date(result.created_at).toLocaleString('ko-KR')}`
  benchmarkElement('execution').textContent = result.results.filter(row => row.method.includes('roi')).map(row => `${row.label}: ${row.execution_summary} (고유 장면 수)`).join(' · ')
  benchmarkElement('preview-caption').textContent = `${context.preview_frame || context.images[0].name} · 같은 장면의 결과입니다. 녹색 경계는 패딩 전 오브젝트 영역입니다. 원본 입력 모드에는 고해상도 정답 영상이 없습니다.`
  const body = benchmarkElement('table-body')
  body.replaceChildren()
  for (const row of result.results) {
    const tr = document.createElement('tr')
    for (const value of [row.label, row.device, row.psnr_exact ? '∞' : benchmarkNumber(row.psnr_db),
      benchmarkNumber(row.ssim, 4), row.roi_psnr_exact ? '∞' : benchmarkNumber(row.roi_psnr_db),
      benchmarkNumber(row.roi_ssim, 4), benchmarkNumber(row.latency_mean_ms), benchmarkNumber(row.latency_p95_ms),
      benchmarkNumber(row.processing_fps), benchmarkNumber(row.gpu_utilization_mean_pct)]) {
      const cell = document.createElement('td')
      cell.textContent = value
      tr.append(cell)
    }
    body.append(tr)
  }
  const charts = benchmarkElement('charts')
  charts.replaceChildren()
  for (const [metric, title, digits] of [['psnr_db', 'PSNR (dB) · 높을수록 좋음', 2],
    ['latency_mean_ms', '처리 지연 (ms) · 낮을수록 좋음', 2],
    ['processing_fps', '프레임 처리량 (FPS)', 1], ['gpu_utilization_mean_pct', 'GPU 전체 사용률 (%)', 1]]) {
    const section = document.createElement('section')
    const heading = document.createElement('h3')
    heading.textContent = title
    section.append(heading)
    const maximum = Math.max(...result.results.map(row => row[metric] ?? 0), 0.001)
    for (const row of result.results) {
      const item = document.createElement('div')
      item.className = 'benchmark-bar-row'
      const label = document.createElement('span')
      label.textContent = row.label
      const bar = document.createElement('meter')
      bar.min = 0
      bar.max = maximum
      bar.value = Number.isFinite(row[metric]) ? Math.max(0, row[metric]) : 0
      bar.setAttribute('aria-label', `${row.label}: ${benchmarkNumber(row[metric], digits)}`)
      const number = document.createElement('strong')
      number.textContent = metric === 'psnr_db' && row.psnr_exact ? '∞' : benchmarkNumber(row[metric], digits)
      item.append(label, bar, number)
      section.append(item)
    }
    charts.append(section)
  }
  const images = benchmarkElement('previews')
  images.replaceChildren()
  for (const preview of previews || []) {
    const figure = document.createElement('figure')
    const image = document.createElement('img')
    image.src = preview.url
    image.alt = preview.label
    const caption = document.createElement('figcaption')
    caption.textContent = preview.label
    figure.append(image, caption)
    images.append(figure)
  }
  const notes = benchmarkElement('notes')
  notes.replaceChildren()
  for (const note of result.protocol.notes) {
    const item = document.createElement('li')
    item.textContent = note
    notes.append(item)
  }
  benchmarkElement('results').hidden = false
}

function renderBenchmarkState (state) {
  if (state.error) { showBenchmarkError(state); return }
  sourceDescription(state)
  const recording = ['recording', 'recording-stop'].includes(state.status)
  benchmarkBusy = ['running', 'stopping'].includes(state.status) || recording
  benchmarkElement('start').disabled = benchmarkBusy
  benchmarkElement('settings').disabled = benchmarkBusy
  document.querySelectorAll('[data-source]').forEach(button => { button.disabled = benchmarkBusy })
  benchmarkElement('record').disabled = benchmarkBusy
  benchmarkElement('record-stop').hidden = !recording
  benchmarkElement('record-stop').disabled = state.status === 'recording-stop'
  benchmarkElement('roi').disabled = !benchmarkRoiAvailable
  benchmarkElement('cancel').hidden = !benchmarkBusy || recording
  benchmarkElement('cancel').disabled = state.status === 'stopping'
  benchmarkElement('progress').hidden = !benchmarkBusy || recording
  benchmarkElement('progress').value = state.progress || 0
  const labels = { idle: '대기', running: '측정 중', stopping: '중단 중', recording: 'DOOM 수집 중', 'recording-stop': '수집 종료 중', completed: '측정 완료', failed: '실패', cancelled: '취소됨' }
  benchmarkElement('status').textContent = labels[state.status] || '대기'
  benchmarkElement('status').dataset.status = state.status
  benchmarkElement('message').textContent = state.message || ''
  if (state.result && benchmarkResultId !== state.result.run_id) {
    renderBenchmarkResults(state.result, state.previews)
    benchmarkResultId = state.result.run_id
  }
  if (state.status === 'running' && !state.result) {
    benchmarkElement('results').hidden = true
    benchmarkResultId = null
  }
  document.querySelectorAll('[data-result]').forEach(button => { button.hidden = !state.result || state.status !== 'completed' })
}

document.querySelectorAll('[data-source]').forEach(button => button.addEventListener('click', async () => {
  try {
    const state = await performanceBridge.select(button.dataset.source)
    if (state.error) showBenchmarkError(state)
    else sourceDescription(state)
  } catch (error) { showBenchmarkError(error) }
}))

benchmarkElement('start').addEventListener('click', async () => {
  const controls = [...benchmarkElement('settings').querySelectorAll('input, select')]
  if (benchmarkBusy || !controls.every(control => control.reportValidity())) return
  try {
    renderBenchmarkState(await performanceBridge.start({
      scale: Number(benchmarkElement('scale').value), device: benchmarkElement('device').value,
      maxFrames: Number(benchmarkElement('frames').value), maxEdge: Number(benchmarkElement('edge').value),
      repeats: Number(benchmarkElement('repeats').value), fp16: benchmarkElement('fp16').checked,
      roi: benchmarkElement('roi').checked, mode: benchmarkElement('mode').value
    }))
  } catch (error) { showBenchmarkError(error) }
})
benchmarkElement('cancel').addEventListener('click', async () => {
  try { renderBenchmarkState(await performanceBridge.cancel()) } catch (error) { showBenchmarkError(error) }
})
benchmarkElement('record').addEventListener('click', async () => {
  try { renderBenchmarkState(await performanceBridge.record()) } catch (error) { showBenchmarkError(error) }
})
benchmarkElement('record-stop').addEventListener('click', async () => {
  try { renderBenchmarkState(await performanceBridge.stopRecording()) } catch (error) { showBenchmarkError(error) }
})
benchmarkElement('mode').addEventListener('change', () => {
  benchmarkElement('edge').disabled = benchmarkElement('mode').value === 'native'
})
document.querySelectorAll('[data-result]').forEach(button => button.addEventListener('click', async () => {
  try {
    const result = await performanceBridge.open(button.dataset.result)
    if (result.error) showBenchmarkError(result)
  } catch (error) { showBenchmarkError(error) }
}))

if (performanceBridge) {
  performanceBridge.onStatus(renderBenchmarkState)
  performanceBridge.getState().then(renderBenchmarkState).catch(showBenchmarkError)
} else {
  benchmarkElement('start').disabled = true
  showBenchmarkError(new Error('성능평가 연결을 사용할 수 없습니다. Avion 앱에서 실행하세요.'))
}
