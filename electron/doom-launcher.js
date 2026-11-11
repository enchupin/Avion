const fs = require('fs')
const path = require('path')
const { spawn } = require('child_process')

const GAME_METHODS = {
  'imdn-roi': 'ROI SR FP32',
  'imdn-roi-fp16': 'ROI SR FP16',
  imdn: '전체 SR FP32',
  'imdn-fp16': '전체 SR FP16',
  bicubic: '바이큐빅',
  bilinear: '바이리니어',
  nearest: '최근접 보간'
}

function validateGameOptions (input = {}, cuda = false) {
  if (!input || typeof input !== 'object' || Array.isArray(input)) throw new Error('게임 실행 설정이 올바르지 않습니다.')
  const method = input.method ?? 'imdn-roi'
  const scale = input.scale ?? 2
  const device = input.device ?? 'auto'
  const debug = input.debug ?? true
  if (typeof method !== 'string' || !Object.hasOwn(GAME_METHODS, method)) throw new Error('지원하지 않는 SR 방식입니다.')
  if (![2, 3, 4].includes(scale)) throw new Error('확대 배율은 ×2, ×3, ×4 중에서 선택하세요.')
  if (!['auto', 'cpu', 'cuda'].includes(device)) throw new Error('지원하지 않는 실행 장치입니다.')
  if (typeof debug !== 'boolean') throw new Error('ROI 영역 표시 설정이 올바르지 않습니다.')
  const model = method.startsWith('imdn')
  const half = method.endsWith('fp16')
  if (model && device === 'cuda' && !cuda) throw new Error('현재 CUDA GPU를 사용할 수 없습니다. SR 실행 환경을 확인하세요.')
  if (half && (!cuda || device === 'cpu')) throw new Error('FP16은 CUDA GPU에서만 실행할 수 있습니다.')
  return { method, scale, device: model ? (device === 'auto' ? (cuda ? 'cuda' : 'cpu') : device) : 'cpu',
    debug: method.includes('roi') && debug }
}

function launchGame ({ gameRoot, srRoot, python, cuda, configDir, options }, spawnProcess = spawn) {
  const selected = validateGameOptions(options, cuda)
  const required = ['run-freedoom2.bat', 'iwads/freedoom-0.13.0/freedoom-0.13.0/freedoom2.wad']
  const missing = required.find(file => !fs.existsSync(path.join(gameRoot, file)))
  const executable = ['build-local/src/Release/chocolate-doom.exe', 'runtime/win-x64/chocolate-doom.exe',
    'build-local/src/Debug/chocolate-doom.exe'].some(file => fs.existsSync(path.join(gameRoot, file)))
  if (missing || !executable) throw new Error(`DOOM 실행 파일이 없습니다: ${missing || 'chocolate-doom.exe'}. Doom 폴더를 확인하세요.`)
  if (!fs.existsSync(path.join(srRoot, 'sr_worker.py'))) throw new Error('SR/sr_worker.py를 찾을 수 없습니다.')
  fs.mkdirSync(configDir, { recursive: true })
  // Each launch uses its selected settings, not inherited SR/recording options.
  const env = Object.fromEntries(Object.entries(process.env).filter(([key]) =>
    !/^AVION_(SR(?:_|$)|ROI_RECORD_|DOOM_)/i.test(key)))
  Object.assign(env, { AVION_DOOM_WAIT: '1', AVION_DOOM_CONFIG_DIR: configDir,
    AVION_SR: '1', AVION_SR_PYTHON: python, AVION_SR_WORKER: path.join(srRoot, 'sr_worker.py'),
    AVION_SR_METHOD: selected.method, AVION_SR_DEVICE: selected.device, AVION_SR_SCALE: String(selected.scale),
    AVION_SR_STATUS_FILE: path.join(configDir, 'worker-status.json'),
    AVION_SR_CHECKPOINT: path.join(srRoot, 'IMDN', 'checkpoints', `IMDN_x${selected.scale}.pth`),
    PYTHONUTF8: '1', PYTHONDONTWRITEBYTECODE: '1' })
  fs.rmSync(env.AVION_SR_STATUS_FILE, { force: true })
  const flags = ['-srnoconsole']
  if (selected.method.includes('roi')) flags.push('-srroi')
  if (selected.debug) flags.push('-srroi-debug')
  // Only fixed flags enter cmd syntax; paths remain in cwd/env.
  return spawnProcess(path.join(process.env.SystemRoot || 'C:\\Windows', 'System32', 'cmd.exe'),
    ['/d', '/s', '/c', ['run-freedoom2.bat', ...flags].join(' ')],
    { cwd: gameRoot, windowsHide: true, env, stdio: ['ignore', 'pipe', 'pipe'] })
}

module.exports = { launchGame, validateGameOptions, GAME_METHODS }
