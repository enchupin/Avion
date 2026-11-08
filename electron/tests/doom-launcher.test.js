const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('fs')
const os = require('os')
const path = require('path')
const { launchRoiGame } = require('../doom-launcher')

function fixture (t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'avion-doom-'))
  t.after(() => fs.rmSync(root, { recursive: true, force: true }))
  const gameRoot = path.join(root, '한글 & spaces (test)', 'Doom')
  for (const file of ['run-freedoom2-roi.bat', 'run-freedoom2.bat', 'runtime/win-x64/chocolate-doom.exe',
    'iwads/freedoom-0.13.0/freedoom-0.13.0/freedoom2.wad']) {
    const target = path.join(gameRoot, file)
    fs.mkdirSync(path.dirname(target), { recursive: true })
    fs.writeFileSync(target, '')
  }
  return { gameRoot, srRoot: path.join(root, 'SR'), configDir: path.join(root, 'user data', 'doom-play'),
    python: path.join(root, 'venv', 'python.exe'), cuda: false }
}

test('ROI launcher uses the fixed batch command and checked Python, with safe path handling', t => {
  const options = fixture(t)
  const result = launchRoiGame(options, (exe, args, config) => ({ exe, args, config }))
  assert.deepEqual(result.args, ['/d', '/s', '/c', 'run-freedoom2-roi.bat'])
  assert.equal(result.config.cwd, options.gameRoot)
  assert.equal(result.config.env.AVION_SR_PYTHON, options.python)
  assert.equal(result.config.env.AVION_SR_DEVICE, 'cpu')
  assert.equal(result.config.env.AVION_DOOM_WAIT, '1')
  assert.equal(result.config.env.AVION_DOOM_CONFIG_DIR, options.configDir)
  assert.ok(fs.existsSync(options.configDir))
  assert.equal(result.config.shell, undefined)
})

test('normal play excludes recording settings and supports an available GPU', t => {
  const options = { ...fixture(t), cuda: true }
  const previous = process.env.AVION_ROI_RECORD_DIR
  process.env.AVION_ROI_RECORD_DIR = 'old-recording'
  try {
    const config = launchRoiGame(options, (_exe, _args, config) => config)
    assert.equal(config.env.AVION_ROI_RECORD_DIR, undefined)
    assert.equal(config.env.AVION_SR_DEVICE, 'auto')
  } finally {
    if (previous === undefined) delete process.env.AVION_ROI_RECORD_DIR
    else process.env.AVION_ROI_RECORD_DIR = previous
  }
})

test('missing packaged assets are reported before starting cmd', t => {
  const options = fixture(t)
  fs.unlinkSync(path.join(options.gameRoot, 'run-freedoom2-roi.bat'))
  assert.throws(() => launchRoiGame(options, () => assert.fail('must not spawn')), /run-freedoom2-roi.bat/)
})
