const { spawn } = require('child_process')
const fs = require('fs')
const path = require('path')
const { randomUUID } = require('crypto')

function validateOptions (options = {}) {
  const integer = (key, fallback, min, max) => {
    const value = options[key] ?? fallback
    if (!Number.isInteger(value) || value < min || value > max) throw new Error(`${key} 설정이 올바르지 않습니다.`)
    return value
  }
  const scale = integer('scale', 2, 2, 4)
  const device = options.device ?? 'auto'
  if (!['auto', 'cpu', 'cuda'].includes(device)) throw new Error('지원하지 않는 실행 장치입니다.')
  if (options.fp16 != null && typeof options.fp16 !== 'boolean') throw new Error('FP16 설정이 올바르지 않습니다.')
  if (options.roi != null && typeof options.roi !== 'boolean') throw new Error('ROI 설정이 올바르지 않습니다.')
  const mode = options.mode ?? 'reconstruction'
  if (!['reconstruction', 'native'].includes(mode)) throw new Error('지원하지 않는 평가 모드입니다.')
  return { scale, device, fp16: Boolean(options.fp16), roi: Boolean(options.roi), mode,
    maxFrames: integer('maxFrames', 5, 1, 100), maxEdge: integer('maxEdge', 512, 32, 4096),
    repeats: integer('repeats', 10, 1, 100) }
}

function registerPerformance ({ app, ipcMain, dialog, shell, BrowserWindow, windows }) {
  const srRoot = app.isPackaged ? path.join(process.resourcesPath, 'SR') : path.resolve(__dirname, '..', 'SR')
  const outputRoot = process.env.AVION_BENCHMARK_OUTPUT || (app.isPackaged
    ? path.join(app.getPath('documents'), 'Avion Benchmarks')
    : path.resolve(__dirname, '..', 'output', 'benchmarks'))
  let selectedSource = null
  let selectedCheckpoint = null
  let child = null
  let doom = null
  let recordingTimer = null
  let cancelRequested = false
  let latestDir = null
  let state = { status: 'idle', progress: 0, message: 'Set5 샘플로 바로 측정하거나 게임 프레임을 선택하세요.' }

  function broadcast (update) {
    state = { ...state, ...update }
    for (const win of windows) {
      if (!win.isDestroyed()) win.webContents.send('performance:status', state)
    }
    return state
  }

  function sourceInfo () {
    let roiAvailable = false
    if (selectedSource) {
      const manifest = fs.statSync(selectedSource).isDirectory() ? path.join(selectedSource, 'roi-manifest.json') : selectedSource
      if (path.basename(manifest) === 'roi-manifest.json' && fs.existsSync(manifest) && fs.statSync(manifest).size < 2 * 1024 * 1024) {
        roiAvailable = JSON.parse(fs.readFileSync(manifest, 'utf8')).format === 'avion.doom.roi.v1'
      }
    }
    return { source: selectedSource || 'Set5 샘플', checkpoint: selectedCheckpoint, roiAvailable }
  }

  function terminate (processToStop) {
    if (process.platform === 'win32' && processToStop.pid) {
      const killer = spawn('taskkill', ['/PID', String(processToStop.pid), '/T', '/F'], { windowsHide: true, stdio: 'ignore' })
      killer.on('error', () => processToStop.kill())
    } else processToStop.kill('SIGINT')
  }

  function stopRecording () {
    if (!doom) return state
    clearInterval(recordingTimer)
    broadcast({ status: 'recording-stop', message: 'DOOM을 종료하고 저장한 장면을 선택합니다.' })
    terminate(doom)
    return state
  }

  function startRecording () {
    if (child || doom) throw new Error('현재 측정 또는 장면 수집을 종료하세요.')
    const gameRoot = app.isPackaged ? path.join(process.resourcesPath, 'Doom') : path.resolve(__dirname, '..', 'Doom')
    const exe = path.join(gameRoot, 'build-local', 'src', 'Release', 'chocolate-doom.exe')
    const iwad = path.join(gameRoot, 'iwads', 'freedoom-0.13.0', 'freedoom-0.13.0', 'freedoom2.wad')
    if (!fs.existsSync(exe) || !fs.existsSync(iwad)) throw new Error('ROI 지원 DOOM 실행 파일 또는 Freedoom2 WAD가 없습니다. README의 DOOM 빌드·설치 절차를 먼저 진행하세요.')
    fs.mkdirSync(outputRoot, { recursive: true })
    const directory = path.join(outputRoot, `doom-scenes-${Date.now()}-${randomUUID().slice(0, 8)}`)
    let stderr = ''
    let failure = null
    const game = spawn(exe, ['-window', '-srroi', '-srcompare', '-srlatency', '-srnoconsole', '-nograbmouse', '-warp', '1', '-skill', '3', '-iwad', iwad,
      '-config', directory + '.cfg', '-extraconfig', directory + '-extra.cfg'], {
      cwd: path.dirname(exe), windowsHide: true, stdio: ['ignore', 'ignore', 'pipe'],
      env: { ...process.env, AVION_SR: '1', AVION_SR_WORKER: path.join(srRoot, 'sr_worker.py'),
        AVION_SR_CHECKPOINT: path.join(srRoot, 'IMDN', 'checkpoints', 'IMDN_x2.pth'), AVION_SR_SCALE: '2',
        AVION_ROI_RECORD_DIR: directory, AVION_ROI_RECORD_INTERVAL_MS: '500', AVION_ROI_RECORD_MAX_FRAMES: '60',
        PYTHONUTF8: '1', PYTHONDONTWRITEBYTECODE: '1' }
    })
    doom = game
    broadcast({ status: 'recording', message: 'DOOM에서 몬스터·아이템이 보이도록 이동하세요. 0.5초마다 최대 60장 저장합니다.', recordedFrames: 0 })
    game.stderr.setEncoding('utf8')
    game.stderr.on('data', data => { stderr = (stderr + data).slice(-3000) })
    game.on('error', error => { failure = `DOOM 실행 실패: ${error.message}` })
    const started = Date.now()
    recordingTimer = setInterval(() => {
      let count = 0
      try { count = JSON.parse(fs.readFileSync(path.join(directory, 'roi-manifest.json'), 'utf8')).frames.length } catch (_) { /* First frame pending. */ }
      broadcast({ recordedFrames: count, message: `DOOM 장면 ${count}/60개 저장 · 이동 후 ‘수집 종료·선택’을 누르세요.` })
      if (count >= 60 || Date.now() - started > 120000) stopRecording()
    }, 500)
    game.on('close', () => {
      if (doom !== game) return
      clearInterval(recordingTimer)
      doom = null
      const manifest = path.join(directory, 'roi-manifest.json')
      if (!failure && fs.existsSync(manifest)) {
        selectedSource = manifest
        broadcast({ status: 'idle', ...sourceInfo(), message: 'DOOM 장면과 ROI를 선택했습니다. 성능 측정을 시작하세요.' })
      } else {
        broadcast({ status: 'failed', message: failure || `저장한 DOOM 프레임이 없습니다. Python 의존성과 ROI 지원 빌드를 확인하세요. ${stderr.trim()}` })
      }
    })
    return state
  }

  function loadResult (directory) {
    const resultPath = path.join(directory, 'report.json')
    if (fs.statSync(resultPath).size > 4 * 1024 * 1024) throw new Error('평가 보고서가 너무 큽니다.')
    const result = JSON.parse(fs.readFileSync(resultPath, 'utf8'))
    if (result.schema_version !== 1 || !Array.isArray(result.results) || !Array.isArray(result.previews)) {
      throw new Error('지원하지 않는 평가 보고서입니다.')
    }
    const previews = result.previews.map((item) => {
      if (!/^preview-(reference|roi-mask|nearest|bilinear|bicubic|imdn|imdn-fp16|imdn-roi|imdn-roi-fp16)\.png$/.test(item.file)) {
        throw new Error('비교 이미지 경로가 올바르지 않습니다.')
      }
      const file = path.join(directory, item.file)
      if (fs.statSync(file).size > 12 * 1024 * 1024) throw new Error('비교 이미지가 너무 큽니다.')
      return { label: item.label, url: `data:image/png;base64,${fs.readFileSync(file).toString('base64')}` }
    })
    return { result, previews }
  }

  // Restore only reports in the application's own output directory.
  if (fs.existsSync(outputRoot)) {
    const entries = fs.readdirSync(outputRoot, { withFileTypes: true })
      .filter(entry => entry.isDirectory() && entry.name.startsWith('bench-'))
      .sort((a, b) => fs.statSync(path.join(outputRoot, b.name)).mtimeMs - fs.statSync(path.join(outputRoot, a.name)).mtimeMs)
    for (const entry of entries) {
      try {
        const directory = path.join(outputRoot, entry.name)
        const loaded = loadResult(directory)
        latestDir = directory
        state = { status: 'completed', progress: 100, message: '최근 측정 결과를 불러왔습니다.', ...loaded }
        break
      } catch (_) { /* Incomplete/cancelled runs have no report.json. */ }
    }
  }

  function start (input) {
    if (doom) throw new Error('GPU 측정 간섭을 막기 위해 DOOM 장면 수집을 먼저 종료하세요.')
    if (child) return { ...state, message: '평가가 이미 실행 중입니다.' }
    const options = validateOptions(input)
    if (options.roi && !sourceInfo().roiAvailable) throw new Error('DOOM 장면을 수집하거나 기존 ROI 데이터를 선택하세요.')
    const script = path.join(srRoot, 'benchmark.py')
    if (!fs.existsSync(script)) throw new Error('SR/benchmark.py를 찾을 수 없습니다.')
    fs.mkdirSync(outputRoot, { recursive: true })
    const directory = path.join(outputRoot, `bench-${new Date().toISOString().replace(/[:.]/g, '-')}-${randomUUID().slice(0, 8)}`)
    const methods = options.roi
      ? ['bicubic', 'imdn', 'imdn-roi', ...(options.fp16 ? ['imdn-fp16', 'imdn-roi-fp16'] : [])]
      : ['nearest', 'bilinear', 'bicubic', 'imdn', ...(options.fp16 ? ['imdn-fp16'] : [])]
    const args = ['-B', '-u', '-X', 'utf8', script, '--output', directory,
      '--scale', String(options.scale), '--device', options.device,
      '--max-frames', String(options.maxFrames), '--max-edge', String(options.maxEdge),
      '--repeats', String(options.repeats), '--mode', options.mode, '--methods', ...methods]
    if (selectedSource) args.push('--source', selectedSource)
    if (selectedCheckpoint) args.push('--checkpoint', selectedCheckpoint)
    const python = process.env.AVION_SR_PYTHON || 'python'
    let buffer = ''
    let stderr = ''
    let completed = false
    let failure = null
    cancelRequested = false
    const processHandle = spawn(python, args, { cwd: srRoot, windowsHide: true,
      env: { ...process.env, PYTHONUTF8: '1', PYTHONDONTWRITEBYTECODE: '1' },
      stdio: ['ignore', 'pipe', 'pipe'] })
    child = processHandle
    broadcast({ status: 'running', message: '평가를 시작합니다.', progress: 0,
      result: null, previews: [], partial: null })
    processHandle.stdout.setEncoding('utf8')
    processHandle.stderr.setEncoding('utf8')
    processHandle.stderr.on('data', data => { stderr = (stderr + data).slice(-6000) })
    processHandle.stdout.on('data', data => {
      if (cancelRequested) return
      buffer += data
      if (buffer.length > 4 * 1024 * 1024) {
        failure = '평가 프로세스 응답이 너무 큽니다.'
        stop()
        return
      }
      let newline
      while ((newline = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, newline).trim()
        buffer = buffer.slice(newline + 1)
        if (!line) continue
        try {
          const event = JSON.parse(line)
          if (event.status === 'completed') {
            // The worker's completed notification is confirmed again on close.
            completed = true
          } else if (event.status === 'failed') {
            failure = event.message
          } else if (event.status === 'running') {
            broadcast({ status: 'running', progress: event.progress,
              message: event.message, partial: event.partial ?? state.partial })
          }
        } catch (error) {
          failure = `평가 응답 해석 실패: ${error.message}`
        }
      }
    })
    processHandle.on('error', error => { failure = `Python 실행 실패: ${error.message}. AVION_SR_PYTHON 설정을 확인하세요.` })
    processHandle.on('close', code => {
      if (child !== processHandle) return
      child = null
      if (cancelRequested) {
        broadcast({ status: failure ? 'failed' : 'cancelled', message: failure || '평가를 취소했습니다.', progress: 0 })
      } else if (code === 0 && completed && !failure) {
        try {
          const loaded = loadResult(directory)
          latestDir = directory
          broadcast({ status: 'completed', progress: 100, message: '측정을 완료했습니다. 보고서와 CSV를 저장했습니다.', ...loaded })
        } catch (error) {
          broadcast({ status: 'failed', message: error.message })
        }
      } else {
        broadcast({ status: 'failed', message: failure || stderr.trim() || `평가 프로세스가 종료되었습니다 (${code}).` })
      }
    })
    return state
  }

  function stop () {
    if (!child) return state
    const processToStop = child
    cancelRequested = true
    broadcast({ status: 'stopping', message: '평가를 중단하고 있습니다.' })
    // Stop this worker and its nvidia-smi child, never unrelated processes.
    terminate(processToStop)
    return state
  }

  async function select (event, type) {
    if (child || doom) throw new Error('평가 또는 장면 수집 종료 후 입력을 변경하세요.')
    if (type === 'sample') {
      selectedSource = null
      return broadcast(sourceInfo())
    }
    if (type === 'default-checkpoint') {
      selectedCheckpoint = null
      return broadcast(sourceInfo())
    }
    const choices = {
      folder: { title: '기준 이미지/프레임 폴더 선택', properties: ['openDirectory'] },
      capture: { title: 'Avion 캡처 선택', properties: ['openFile'], filters: [{ name: 'Avion Capture', extensions: ['json', 'bgra', 'csv'] }] },
      roi: { title: 'DOOM roi-manifest.json 선택', properties: ['openFile'], filters: [{ name: 'DOOM ROI manifest', extensions: ['json'] }] },
      checkpoint: { title: 'IMDN 가중치 선택 (확대 배율 일치 필요)', properties: ['openFile'], filters: [{ name: 'PyTorch weights', extensions: ['pth', 'pt'] }] }
    }
    if (!choices[type]) throw new Error('지원하지 않는 입력 종류입니다.')
    const owner = BrowserWindow.fromWebContents(event.sender)
    const selected = owner ? await dialog.showOpenDialog(owner, choices[type]) : await dialog.showOpenDialog(choices[type])
    if (!selected.canceled && selected.filePaths[0]) {
      if (type === 'checkpoint') selectedCheckpoint = selected.filePaths[0]
      else selectedSource = selected.filePaths[0]
    }
    return broadcast(sourceInfo())
  }

  const guarded = handler => async (...args) => {
    try { return await handler(...args) } catch (error) { return { error: error.message } }
  }
  ipcMain.handle('performance:get-state', guarded(() => ({ ...state, ...sourceInfo() })))
  ipcMain.handle('performance:record', guarded(() => startRecording()))
  ipcMain.handle('performance:record-stop', guarded(() => stopRecording()))
  ipcMain.handle('performance:select', guarded(select))
  ipcMain.handle('performance:start', guarded((_event, options) => start(options)))
  ipcMain.handle('performance:cancel', () => stop())
  ipcMain.handle('performance:open', guarded(async (_event, kind) => {
    if (!latestDir) throw new Error('저장한 평가 결과가 없습니다.')
    if (!['report', 'folder', 'csv'].includes(kind)) throw new Error('지원하지 않는 결과 형식입니다.')
    const target = kind === 'folder' ? latestDir : path.join(latestDir, kind === 'report' ? 'report.html' : 'summary.csv')
    const message = await shell.openPath(target)
    if (message) throw new Error(message)
    return { success: true }
  }))
  app.on('before-quit', stop)
  app.on('before-quit', stopRecording)
  return { start, stop, startRecording, stopRecording, getState: () => state }
}

module.exports = { registerPerformance, validateOptions }
