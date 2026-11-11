const { spawn } = require('child_process')
const fs = require('fs')
const path = require('path')
const { launchGame, validateGameOptions, GAME_METHODS } = require('./doom-launcher')

function registerDoomGame ({ app, ipcMain, windows, environment }) {
  const srRoot = app.isPackaged ? path.join(process.resourcesPath, 'SR') : path.resolve(__dirname, '..', 'SR')
  let gameProcess = null
  let gameStopRequested = false
  let state = { status: 'idle', gameReady: false, message: 'SR 방식을 선택하고 DOOM을 실행하세요.' }

  function broadcast (update) {
    state = { ...state, ...update }
    for (const win of windows) {
      if (!win.isDestroyed()) win.webContents.send('doom:status', state)
    }
    return state
  }

  function terminate (processToStop) {
    if (process.platform === 'win32' && processToStop.pid) {
      const killer = spawn('taskkill', ['/PID', String(processToStop.pid), '/T', '/F'], { windowsHide: true, stdio: 'ignore' })
      killer.on('error', () => processToStop.kill())
      return killer
    } else processToStop.kill('SIGINT')
  }

  function play (input) {
    if (process.platform !== 'win32') throw new Error('DOOM 실행은 Windows에서 지원합니다.')
    if (gameProcess) throw new Error('실행 중인 DOOM을 먼저 종료하세요.')
    const python = environment.requirePython()
    const options = validateGameOptions(input, environment.getState().report?.cuda)
    const configDir = path.join(app.getPath('userData'), 'doom-play')
    const description = `${GAME_METHODS[options.method]} · ×${options.scale} · ${options.device === 'cuda' ? 'CUDA GPU' : 'CPU'}`
    const game = launchGame({
      gameRoot: app.isPackaged ? path.join(process.resourcesPath, 'Doom') : path.resolve(__dirname, '..', 'Doom'),
      srRoot, python, cuda: environment.getState().report?.cuda,
      configDir, options
    })
    gameProcess = game
    gameStopRequested = false
    let failure = null
    let output = ''
    let ready = false
    // The native host silences Python stderr. Read the worker's separate status
    // channel so a model error cannot masquerade as successful SR gameplay.
    const checkWorker = (closing = false) => {
      if (gameStopRequested || failure) return
      let worker
      try { worker = JSON.parse(fs.readFileSync(path.join(configDir, 'worker-status.json'), 'utf8')) } catch (_) { return }
      if (worker.status === 'failed') {
        failure = `${description} 실행 실패: ${worker.message}`
        broadcast({ message: failure })
        if (!closing) terminate(game)
      } else if (worker.status === 'ready' && !ready) {
        ready = true
        broadcast({ gameReady: true, message: `${description} 적용 완료 · SR 입력 320×200 (1:1) / SR 결과 ${320 * options.scale}×${200 * options.scale}` })
      }
    }
    const timer = setInterval(() => checkWorker(), 500)
    for (const stream of [game.stdout, game.stderr]) {
      stream.setEncoding('utf8')
      stream.on('data', data => {
        output = (output + data).slice(-3000)
        if (!gameStopRequested && !failure && /SR: (worker communication failed|failed to start worker|worker script was not found)/.test(output)) {
          checkWorker()
          if (!failure) {
            failure = `${description}: SR worker를 실행할 수 없습니다. SR 실행 환경을 다시 확인하세요.`
            terminate(game)
          }
        }
      })
    }
    game.on('error', error => { failure = error.message })
    game.on('close', code => {
      clearInterval(timer)
      if (gameProcess !== game) return
      checkWorker(true)
      gameProcess = null
      const failed = failure || (!gameStopRequested && code !== 0)
      broadcast({ status: failed ? 'failed' : 'idle', gameReady: false,
        message: failure || (failed ? `DOOM 실행 실패 (${code}): ${output.trim()}` : 'DOOM을 종료했습니다. SR 방식을 변경하고 다시 실행할 수 있습니다.') })
    })
    return broadcast({ status: 'playing', gameOptions: options, gameReady: false,
      message: `${description} 준비 중 · DOOM과 SR worker를 시작합니다.` })
  }

  function stopGame () {
    if (!gameProcess || gameStopRequested) return state
    gameStopRequested = true
    broadcast({ status: 'playing-stop', message: 'DOOM과 SR worker를 종료합니다.' })
    terminate(gameProcess)
    return state
  }

  const guarded = handler => async (...args) => {
    try { return await handler(...args) } catch (error) { return { error: error.message } }
  }
  ipcMain.handle('doom:get-state', () => state)
  ipcMain.handle('doom:play', guarded((_event, options) => play(options)))
  ipcMain.handle('doom:stop', guarded(stopGame))
  let quittingGame = false
  app.on('before-quit', event => {
    if (!gameProcess || quittingGame) return
    // Keep the cmd parent alive until taskkill has enumerated its descendants.
    // Exiting Electron first can orphan the game launched by START /WAIT.
    event.preventDefault()
    quittingGame = true
    const killer = terminate(gameProcess)
    if (killer) killer.once('close', () => app.quit())
    else app.quit()
  })
  return { getState: () => state, isRunning: () => Boolean(gameProcess) }
}

module.exports = { registerDoomGame }
