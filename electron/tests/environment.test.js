const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('fs')
const os = require('os')
const path = require('path')
const { createEnvironmentManager, pathsFor, verifyArchive } = require('../sr-environment')

function fixture (t, extra = {}) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'avion-env-test-'))
  t.after(() => fs.rmSync(root, { recursive: true, force: true }))
  const options = { isPackaged: true, userData: path.join(root, 'user data'), resourcesPath: path.join(root, 'resources'), repoRoot: root, ...extra }
  const locations = pathsFor(options)
  const installed = new Set()
  const calls = []
  let python = false
  const run = async (exe, args, opts) => {
    calls.push({ exe, args, opts })
    if (args.includes('-c') && args.at(-1).includes('AVION_PYTHON:')) {
      if (exe === locations.basePython || (python && exe === locations.python)) return 'AVION_PYTHON:' + JSON.stringify({ python: exe, compatible: true })
      throw new Error('Python not installed')
    }
    if (args.some(a => a.endsWith('sr-probe-bootstrap.py'))) {
      const packages = Object.fromEntries(['numpy', 'cv2', 'skimage', 'torch'].map(key => [key, installed.has(key) ? '9.0' : null]))
      const ready = Object.values(packages).every(Boolean)
      return JSON.stringify({ python: exe, version: '3.13.17', compatible: true, isVenv: true, packages, ready, model: ready, cuda: false, errors: [] })
    }
    if (args.includes('venv')) python = true
    const mapping = { 'numpy>=1.24': 'numpy', 'opencv-python>=4.8': 'cv2', 'scikit-image>=0.21': 'skimage', 'torch>=2.1': 'torch' }
    if (args.includes('install')) installed.add(mapping[args.at(-1)])
    return ''
  }
  const manager = createEnvironmentManager(options, { run, platform: 'win32', arch: 'x64' })
  return { manager, calls, options, locations, installed, run }
}

test('packaged installs use writable user data, never the app resources directory', t => {
  const { locations, options } = fixture(t)
  assert.ok(locations.python.startsWith(options.userData + path.sep))
  assert.ok(locations.srRoot.startsWith(options.resourcesPath + path.sep))
})

test('check is read-only; installation requires checking and accepts only fixed components', async t => {
  const { manager, calls } = fixture(t)
  assert.match((await manager.install('python')).error, /먼저/)
  assert.match((await manager.install('torch; calc')).error, /지원하지/)
  assert.equal((await manager.check()).status, 'missing')
  assert.ok(calls.every(call => call.args.includes('-c')))
  assert.throws(() => manager.requirePython(), /먼저/)
})

test('missing Python and individual packages install sequentially and are rechecked', async t => {
  const { manager, calls, locations } = fixture(t)
  await manager.check()
  const python = await manager.install('python')
  assert.equal(python.report.version, '3.13.17')
  assert.equal(python.ready, false)
  assert.equal(calls.filter(call => call.args.includes('install')).length, 0)
  for (const item of ['numpy', 'cv2', 'skimage', 'torch']) await manager.install(item)
  assert.equal(manager.getState().ready, true)
  assert.equal(manager.requirePython(), locations.python)
  const torch = calls.find(call => call.args.at(-1) === 'torch>=2.1')
  assert.ok(torch.args.includes('https://download.pytorch.org/whl/cpu'))
  assert.equal(torch.exe, locations.python)
})

test('setup cannot run while a measurement or recording is using Python', async t => {
  const { manager, calls } = fixture(t, { isInUse: () => true })
  assert.match((await manager.check()).error, /종료/)
  assert.match((await manager.install('python')).error, /종료/)
  assert.equal(calls.length, 0)
})

test('duplicate clicks share one operation and cancellation leaves a retryable state', async t => {
  const { options } = fixture(t)
  let count = 0
  const manager = createEnvironmentManager(options, { platform: 'win32', arch: 'x64',
    run: (_exe, _args, { signal }) => new Promise((_resolve, reject) => {
      count++
      signal.addEventListener('abort', () => reject(new Error('cancelled')), { once: true })
    }) })
  const first = manager.check()
  const second = manager.check()
  assert.equal(first, second)
  manager.cancel()
  assert.equal((await first).status, 'cancelled')
  assert.equal(count, 1)
  assert.equal(manager.isBusy(), false)
})

test('download failure never reports a usable environment and can be retried', async t => {
  const { options, run, locations } = fixture(t)
  const manager = createEnvironmentManager(options, { platform: 'win32', arch: 'x64',
    run: (exe, args, opts) => exe === locations.basePython ? Promise.reject(new Error('missing')) : run(exe, args, opts),
    downloadPython: async () => { throw new Error('network offline') } })
  await manager.check()
  assert.equal((await manager.install('python')).status, 'failed')
  assert.match(manager.getState().message, /network offline/)
  assert.equal(manager.isBusy(), false)
  assert.throws(() => manager.requirePython())
})

test('corrupt downloads fail hash validation before extraction or execution', async t => {
  const { options } = fixture(t)
  const file = path.join(options.repoRoot, 'corrupt.zip')
  fs.writeFileSync(file, 'not a Python archive')
  await assert.rejects(verifyArchive(file), /SHA-256/)
})
