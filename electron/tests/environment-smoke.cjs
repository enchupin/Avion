// Run from electron/: electron tests/environment-smoke.cjs
const { app, BrowserWindow } = require('electron')
const path = require('path')
const fs = require('fs')
const assert = require('node:assert/strict')
const output = path.resolve(__dirname, '../../.avion-runtime/ui-test')
fs.mkdirSync(output, { recursive: true })
app.setPath('userData', output)
process.env.AVION_TEST_HEADLESS = '1'
require('../index')
const deadline = setTimeout(() => { console.error('Electron environment check timed out'); app.exit(1) }, 180000)
app.whenReady().then(async () => {
  try {
    const win = BrowserWindow.getAllWindows()[0]
    await new Promise(resolve => win.webContents.once('did-finish-load', resolve))
    const initial = await win.webContents.executeJavaScript(`({
      label: document.getElementById('sr-check').textContent,
      installs: document.querySelectorAll('#sr-components button').length,
      blocked: document.getElementById('benchmark-start').disabled,
      playBlocked: document.getElementById('benchmark-play').disabled
    })`)
    assert.equal(initial.label, '설치 여부 확인')
    assert.equal(initial.installs, 0)
    assert.equal(initial.blocked, true)
    assert.equal(initial.playBlocked, true)
    const missing = await win.webContents.executeJavaScript(`(() => {
      showEnvironment({ status: 'missing', ready: false });
      const withoutPython = [...document.querySelectorAll('#sr-components button')].map(button => button.disabled);
      showEnvironment({ status: 'missing', ready: false, report: { version: '3.13.17', packages: { numpy: '2.5.4' } } });
      const withPython = [...document.querySelectorAll('#sr-components button')].map(button => ({ label: button.textContent, disabled: button.disabled }));
      return { withoutPython, withPython };
    })()`)
    assert.deepEqual(missing.withoutPython, [false, true, true, true, true])
    assert.equal(missing.withPython.length, 3)
    assert.ok(missing.withPython.every(button => !button.disabled))
    const report = await win.webContents.executeJavaScript(`new Promise(resolve => {
      const unsubscribe = window.srEnvironment.onStatus(state => {
        if (['ready', 'missing', 'failed'].includes(state.status)) { unsubscribe(); resolve(state) }
      });
      document.getElementById('sr-check').click();
    })`)
    assert.equal(report.ready, true, JSON.stringify(report))
    const rendered = await win.webContents.executeJavaScript(`({
      rows: document.getElementById('sr-components').textContent,
      gpu: document.getElementById('sr-gpu').textContent,
      blocked: document.getElementById('benchmark-start').disabled,
      cudaDisabled: document.querySelector('#benchmark-device option[value="cuda"]').disabled,
      playBlocked: document.getElementById('benchmark-play').disabled
    })`)
    assert.match(rendered.rows, /설치됨/)
    assert.equal(rendered.blocked, false)
    assert.equal(rendered.playBlocked, false)
    assert.equal(rendered.cudaDisabled, !report.report.cuda)
    if (process.env.AVION_TEST_ROI_LAUNCH === '1') {
      const playing = await win.webContents.executeJavaScript(`new Promise(resolve => {
        const unsubscribe = window.performanceEvaluation.onStatus(state => {
          if (['playing', 'failed'].includes(state.status)) { unsubscribe(); resolve(state) }
        });
        document.getElementById('benchmark-play').click();
      })`)
      assert.equal(playing.status, 'playing')
      await new Promise(resolve => setTimeout(resolve, 5000))
      const active = await win.webContents.executeJavaScript(`(async () => ({
        state: await window.performanceEvaluation.getState(),
        duplicate: await window.performanceEvaluation.play(),
        setup: await window.srEnvironment.check(),
        disabled: document.getElementById('benchmark-play').disabled,
        label: document.getElementById('benchmark-play').textContent
      }))()`)
      assert.equal(active.state.status, 'playing', JSON.stringify(active))
      assert.ok(active.duplicate.error)
      assert.ok(active.setup.error)
      assert.equal(active.disabled, true)
      assert.equal(active.label, 'ROI 게임 실행 중')
      console.log('ROI batch launch and busy-state guards passed')
    }
    console.log(JSON.stringify({ status: 'passed', ...rendered }))
    clearTimeout(deadline)
    app.quit()
  } catch (error) {
    console.error(error)
    clearTimeout(deadline)
    process.exitCode = 1
    app.quit()
  }
})
