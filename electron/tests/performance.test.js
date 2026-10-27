const test = require('node:test')
const assert = require('node:assert/strict')
const { validateOptions } = require('../performance')

test('IPC rejects invalid numeric, device and precision parameters', () => {
  for (const options of [{ scale: '2' }, { scale: 5 }, { repeats: Infinity },
    { maxFrames: 101 }, { maxFrames: -1 }, { maxEdge: 100000 },
    { device: 'cuda; echo injected' }, { fp16: 'true' }, { repeats: 0 }, { roi: 'true' }, { mode: 'fake-gt' }]) {
    assert.throws(() => validateOptions(options))
  }
})

test('ROI and native evaluation are explicit validated choices', () => {
  const options = validateOptions({ roi: true, mode: 'native', fp16: true })
  assert.equal(options.roi, true)
  assert.equal(options.mode, 'native')
  assert.equal(validateOptions().roi, false)
  assert.equal(validateOptions().mode, 'reconstruction')
})

test('valid options remain numeric arguments with no source/path passthrough', () => {
  const options = validateOptions({ scale: 4, device: 'cuda', fp16: true,
    maxFrames: 10, maxEdge: 960, repeats: 20, source: 'arbitrary-path' })
  assert.equal(options.scale, 4)
  assert.equal(options.fp16, true)
  assert.equal(options.source, undefined)
})
