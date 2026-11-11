// Real Electron -> IPC -> batch -> DOOM -> Python roundtrip, using SDL's hidden renderer.
// Run: electron tests/doom-smoke.cjs
const { app, BrowserWindow } = require('electron')
const path = require('path')
const fs = require('fs')
const assert = require('node:assert/strict')
const output = path.resolve(__dirname, '../../.avion-runtime/doom-ui-test')
fs.mkdirSync(output, { recursive: true })
app.setPath('userData', output)
Object.assign(process.env, { AVION_TEST_HEADLESS: '1', SDL_VIDEODRIVER: 'dummy',
  SDL_RENDER_DRIVER: 'software', SDL_AUDIODRIVER: 'dummy' })
require('../index')
const deadline = setTimeout(() => { console.error('DOOM integration timed out'); process.exitCode = 1; app.quit() }, 180000)
const delay = ms => new Promise(resolve => setTimeout(resolve, ms))

app.whenReady().then(async () => {
  const win = BrowserWindow.getAllWindows()[0]
  win.webContents.setBackgroundThrottling(false)
  const js = code => win.webContents.executeJavaScript(code)
  const waitState = async (predicate, allowFailure = false) => {
    for (let i = 0; i < 120; i++) {
      const state = await js('window.doomGame.getState()')
      if (state.status === 'failed' && !allowFailure) assert.fail(state.message)
      if (predicate(state)) return state
      await delay(250)
    }
    assert.fail('Game status timed out')
  }
  try {
    await new Promise(resolve => win.webContents.once('did-finish-load', resolve))
    const errors = []
    win.webContents.on('console-message', details => { if (details.level === 'error') errors.push(details.message) })
    assert.equal(await js("document.getElementById('doom-play').disabled"), true)
    const environment = await js('window.srEnvironment.check()')
    assert.equal(environment.ready, true, JSON.stringify(environment))
    if (process.env.AVION_TEST_DOOM_UI_ONLY === '1') {
      win.setContentSize(1280, 1400)
      await delay(250)
      await win.webContents.capturePage(undefined, { stayHidden: true, stayAwake: true })
      await delay(250)
      fs.writeFileSync(path.join(output, 'doom-launcher-layout.png'),
        (await win.webContents.capturePage(undefined, { stayHidden: true, stayAwake: true })).toPNG())
      console.log('DOOM launcher layout captured')
      return
    }
    const cpuUI = await js(`(() => {
      showEnvironment({ status: 'ready', ready: true, report: { cuda: false } });
      return { half: [...document.querySelectorAll('#doom-method option')].filter(o => o.value.endsWith('fp16')).every(o => o.disabled),
        gpu: document.querySelector('#doom-device option[value="cuda"]').disabled };
    })()`)
    assert.deepEqual(cpuUI, { half: true, gpu: true })
    await js('window.srEnvironment.getState().then(showEnvironment)')
    const invalid = await js("window.doomGame.play({method:'imdn-fp16',device:'cpu'})")
    assert.match(invalid.error, /CUDA/)
    assert.equal((await js('window.doomGame.getState()')).status === 'playing', false)
    const methods = ['imdn-roi', 'imdn', 'bicubic', 'bilinear', 'nearest',
      ...(environment.report.cuda ? ['imdn-roi-fp16', 'imdn-fp16'] : [])]
    for (const method of methods) {
      const scale = method === 'bicubic' ? 3 : method === 'nearest' ? 4 : 2
      await js(`(() => {
        document.getElementById('doom-method').value = ${JSON.stringify(method)};
        document.getElementById('doom-scale').value = '${scale}';
        document.getElementById('doom-method').dispatchEvent(new Event('change'));
        document.getElementById('doom-play').click();
      })()`)
      const state = await waitState(state => state.status === 'playing' && state.gameReady)
      assert.equal(state.gameOptions.method, method)
      assert.equal(state.gameOptions.scale, scale)
      const actual = JSON.parse(fs.readFileSync(path.join(output, 'doom-play/worker-status.json'), 'utf8'))
      assert.equal(actual.status, 'ready')
      assert.equal(actual.method, method)
      assert.equal(actual.scale, scale)
      const busy = await js(`(async () => ({
        duplicate: await window.doomGame.play(),
        setup: await window.srEnvironment.check(),
        settings: document.getElementById('doom-settings').disabled,
        stop: document.getElementById('doom-stop').hidden,
        environment: document.getElementById('sr-check').disabled
      }))()`)
      assert.ok(busy.duplicate.error)
      assert.ok(busy.setup.error)
      assert.equal(busy.settings, true)
      assert.equal(busy.stop, false)
      assert.equal(busy.environment, true)
      if (method === 'imdn-roi') {
        await js("document.getElementById('doom-launcher-title').scrollIntoView({block:'start'})")
        await delay(150)
        fs.writeFileSync(path.join(output, 'doom-launcher.png'),
          (await win.webContents.capturePage(undefined, { stayHidden: true, stayAwake: true })).toPNG())
      }
      await js("document.getElementById('doom-stop').click()")
      await waitState(state => state.status === 'idle')
      assert.equal(await js("document.getElementById('doom-settings').disabled"), false)
      assert.equal(await js("document.getElementById('doom-play').disabled"), false)
      console.log(`PASS ${method} x${scale}: ${actual.device}; worker ready, controls locked, clean stop`)
    }
    // Test supervision separately from Python's actual model-failure tests.
    await js("window.doomGame.play({method:'bicubic'})")
    await waitState(state => state.status === 'playing' && state.gameReady)
    const statusFile = path.join(output, 'doom-play/worker-status.json')
    fs.writeFileSync(statusFile, JSON.stringify({ status: 'failed', message: 'integration worker failure' }))
    const failed = await waitState(state => state.status === 'failed', true)
    assert.match(failed.message, /integration worker failure/)
    assert.equal(await js("document.getElementById('doom-play').disabled"), false)
    console.log('PASS worker failure: game stopped, error displayed, launch unlocked')
    assert.deepEqual(errors, [])
    console.log('Electron DOOM launcher integration passed')
  } catch (error) {
    console.error(error)
    process.exitCode = 1
  } finally {
    clearTimeout(deadline)
    app.quit()
  }
})
