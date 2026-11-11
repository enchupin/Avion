const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('fs')
const os = require('os')
const path = require('path')
const { launchGame, validateGameOptions, GAME_METHODS } = require('../doom-launcher')

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
  fs.mkdirSync(path.join(root, 'SR'))
  fs.writeFileSync(path.join(root, 'SR', 'sr_worker.py'), '')
  return { gameRoot, srRoot: path.join(root, 'SR'), configDir: path.join(root, 'user data', 'doom-play'),
    python: path.join(root, 'venv', 'python.exe'), cuda: false }
}

test('default ROI launcher uses fixed flags and checked Python, with safe path handling', t => {
  const options = fixture(t)
  const result = launchGame(options, (exe, args, config) => ({ exe, args, config }))
  assert.deepEqual(result.args, ['/d', '/s', '/c', 'run-freedoom2.bat -srnoconsole -srroi -srroi-debug'])
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
    const config = launchGame(options, (_exe, _args, config) => config)
    assert.equal(config.env.AVION_ROI_RECORD_DIR, undefined)
    assert.equal(config.env.AVION_SR_DEVICE, 'cuda')
  } finally {
    if (previous === undefined) delete process.env.AVION_ROI_RECORD_DIR
    else process.env.AVION_ROI_RECORD_DIR = previous
  }
})

test('missing packaged assets are reported before starting cmd', t => {
  const options = fixture(t)
  fs.unlinkSync(path.join(options.gameRoot, 'run-freedoom2.bat'))
  assert.throws(() => launchGame(options, () => assert.fail('must not spawn')), /run-freedoom2.bat/)
})

test('all SR choices propagate to the worker with matching scale, ROI flags and device', t => {
  const fixtureOptions = fixture(t)
  for (const method of Object.keys(GAME_METHODS)) {
    for (const scale of [2, 3, 4]) {
      const result = launchGame({ ...fixtureOptions, cuda: true, options: { method, scale, debug: false } },
        (exe, args, config) => ({ args, ...config }))
      assert.equal(result.env.AVION_SR_METHOD, method)
      assert.equal(result.env.AVION_SR_SCALE, String(scale))
      assert.ok(result.env.AVION_SR_CHECKPOINT.endsWith(`IMDN_x${scale}.pth`))
      assert.equal(result.env.AVION_SR_DEVICE, method.startsWith('imdn') ? 'cuda' : 'cpu')
      assert.equal(result.args.at(-1).includes('-srroi'), method.includes('roi'))
      assert.ok(!result.args.at(-1).includes('-srroi-debug'))
    }
  }
})

test('invalid IPC options and unsupported GPU/FP16 choices are rejected before spawn', t => {
  const options = fixture(t)
  for (const input of [null, [], { method: 'bicubic & calc' }, { method: '__proto__' }, { method: ['imdn'] },
    { device: 'cuda' }, { method: 'imdn-fp16' }, { method: 'imdn-roi-fp16' },
    { scale: '2' }, { scale: 1 }, { debug: 'true' }]) {
    assert.throws(() => launchGame({ ...options, options: input }, () => assert.fail('must not spawn')))
  }
  assert.throws(() => validateGameOptions({ method: 'imdn-fp16', device: 'cpu' }, true), /CUDA/)
  assert.equal(validateGameOptions({ method: 'bicubic', device: 'cuda' }).device, 'cpu')
})

test('a new launch clears stale worker status', t => {
  const options = fixture(t)
  fs.mkdirSync(options.configDir, { recursive: true })
  const status = path.join(options.configDir, 'worker-status.json')
  fs.writeFileSync(status, '{"status":"ready"}')
  launchGame(options, (_exe, _args, config) => {
    assert.equal(config.env.AVION_SR_STATUS_FILE, status)
    assert.equal(fs.existsSync(status), false)
  })
})
