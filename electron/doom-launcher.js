const fs = require('fs')
const path = require('path')
const { spawn } = require('child_process')

function launchRoiGame ({ gameRoot, srRoot, python, cuda, configDir }, spawnProcess = spawn) {
  const required = ['run-freedoom2-roi.bat', 'run-freedoom2.bat',
    'iwads/freedoom-0.13.0/freedoom-0.13.0/freedoom2.wad']
  const missing = required.find(file => !fs.existsSync(path.join(gameRoot, file)))
  const executable = ['build-local/src/Release/chocolate-doom.exe', 'runtime/win-x64/chocolate-doom.exe',
    'build-local/src/Debug/chocolate-doom.exe'].some(file => fs.existsSync(path.join(gameRoot, file)))
  if (missing || !executable) throw new Error(`ROI 게임 실행 파일이 없습니다: ${missing || 'chocolate-doom.exe'}. Doom 폴더를 확인하세요.`)
  fs.mkdirSync(configDir, { recursive: true })
  const env = { ...process.env, AVION_DOOM_WAIT: '1', AVION_DOOM_CONFIG_DIR: configDir,
    AVION_SR: '1', AVION_SR_PYTHON: python, AVION_SR_WORKER: path.join(srRoot, 'sr_worker.py'),
    AVION_SR_DEVICE: cuda ? 'auto' : 'cpu', AVION_SR_SCALE: '2',
    AVION_SR_CHECKPOINT: path.join(srRoot, 'IMDN', 'checkpoints', 'IMDN_x2.pth'),
    PYTHONUTF8: '1', PYTHONDONTWRITEBYTECODE: '1' }
  // Playing from this button must not inherit a previous scene-recording session.
  for (const key of Object.keys(env)) if (key.startsWith('AVION_ROI_RECORD_')) delete env[key]
  // The command is fixed. All filesystem paths stay in cwd/env, outside cmd syntax.
  return spawnProcess(path.join(process.env.SystemRoot || 'C:\\Windows', 'System32', 'cmd.exe'),
    ['/d', '/s', '/c', 'run-freedoom2-roi.bat'],
    { cwd: gameRoot, windowsHide: true, env, stdio: ['ignore', 'pipe', 'pipe'] })
}

module.exports = { launchRoiGame }
