const fs = require('fs')
const path = require('path')
const https = require('https')
const { spawn } = require('child_process')
const { createHash, randomUUID } = require('crypto')
const { pipeline } = require('stream/promises')
const { Transform } = require('stream')

// Official CPython NuGet distribution; includes venv/ensurepip, requires no installer or registry changes.
// https://docs.python.org/3.13/using/windows.html#the-nuget-org-packages
const PYTHON_VERSION = '3.13.17'
const PYTHON_URL = `https://api.nuget.org/v3-flatcontainer/python/${PYTHON_VERSION}/python.${PYTHON_VERSION}.nupkg`
const PYTHON_SHA256 = 'dce409744b088f94c6e37ba41f8100c384c27e029e843237364c602d30d475e5'
const COMPONENTS = { python: null, numpy: 'numpy>=1.24', cv2: 'opencv-python>=4.8', skimage: 'scikit-image>=0.21', torch: 'torch>=2.1' }
const PROBE_PREFIX = 'AVION_PYTHON:'
const PROBE = `import sys,struct,json; print('${PROBE_PREFIX}'+json.dumps({'python':sys.executable,'compatible':(3,10)<=sys.version_info[:2]<(3,14) and struct.calcsize('P')==8}))`

function pathsFor ({ isPackaged, userData, resourcesPath, repoRoot }) {
  const root = isPackaged ? path.join(userData, 'sr-runtime') : path.join(repoRoot, '.avion-runtime')
  return { root, srRoot: path.join(isPackaged ? resourcesPath : repoRoot, 'SR'),
    python: path.join(root, 'venv', 'Scripts', 'python.exe'),
    basePython: path.join(root, `python-${PYTHON_VERSION}`, 'tools', 'python.exe'),
    projectPython: path.join(repoRoot, '.venv', 'Scripts', 'python.exe') }
}

function terminate (child) {
  if (process.platform === 'win32' && child.pid) {
    const killer = spawn('taskkill.exe', ['/PID', String(child.pid), '/T', '/F'], { windowsHide: true, stdio: 'ignore' })
    killer.on('error', () => child.kill())
  } else child.kill()
}

function run (command, args, { timeout = 120000, onOutput = () => {}, signal, env = {} } = {}) {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(new Error('설치가 취소되었습니다.'))
    const child = spawn(command, args, { windowsHide: true, shell: false,
      env: { ...process.env, PYTHONUTF8: '1', PYTHONDONTWRITEBYTECODE: '1',
        PIP_DISABLE_PIP_VERSION_CHECK: '1', PIP_NO_INPUT: '1', ...env }, stdio: ['ignore', 'pipe', 'pipe'] })
    let stdout = '', stderr = '', failure = null
    const stop = message => { failure = new Error(message); terminate(child) }
    const abort = () => stop('설치가 취소되었습니다. 다시 실행하면 이어서 점검합니다.')
    signal?.addEventListener('abort', abort, { once: true })
    const timer = setTimeout(() => stop('작업 시간이 초과되었습니다. 네트워크와 설치 로그를 확인하고 다시 시도하세요.'), timeout)
    child.stdout.setEncoding('utf8')
    child.stderr.setEncoding('utf8')
    child.stdout.on('data', data => { stdout = (stdout + data).slice(-65536); onOutput(data) })
    child.stderr.on('data', data => { stderr = (stderr + data).slice(-65536); onOutput(data) })
    child.on('error', error => { failure = error })
    child.on('close', code => {
      clearTimeout(timer)
      signal?.removeEventListener('abort', abort)
      if (failure) reject(failure)
      else if (code !== 0) reject(new Error((stderr || stdout).trim().slice(-3000) || `프로세스 종료 코드: ${code}`))
      else resolve(stdout)
    })
  })
}

async function verifyArchive (file) {
  const hash = createHash('sha256')
  for await (const chunk of fs.createReadStream(file)) hash.update(chunk)
  if (hash.digest('hex') !== PYTHON_SHA256) throw new Error('Python 다운로드의 SHA-256 검증에 실패했습니다. 다시 설치해 주세요.')
}

async function downloadPython (file, { signal, onProgress }) {
  const response = await new Promise((resolve, reject) => {
    const request = https.get(PYTHON_URL, { signal }, response => {
      if (response.statusCode !== 200) {
        response.resume()
        reject(new Error(`Python 다운로드 실패: HTTP ${response.statusCode}`))
      } else resolve(response)
    })
    request.setTimeout(60000, () => request.destroy(new Error('Python 다운로드 연결 시간이 초과되었습니다.')))
    request.on('error', reject)
  })
  const total = Number(response.headers['content-length'])
  let received = 0, lastUpdate = 0
  const meter = new Transform({ transform (chunk, encoding, callback) {
    received += chunk.length
    if (received > 100 * 1024 * 1024) return callback(new Error('Python 다운로드 크기가 예상 범위를 초과했습니다.'))
    if (Date.now() - lastUpdate > 300) {
      onProgress(total ? Math.round(received / total * 100) : null)
      lastUpdate = Date.now()
    }
    callback(null, chunk)
  } })
  await pipeline(response, meter, fs.createWriteStream(file), { signal })
  await verifyArchive(file)
}

function createEnvironmentManager (options, dependencies = {}) {
  const locations = pathsFor(options)
  const execute = dependencies.run || run
  const download = dependencies.downloadPython || downloadPython
  const platform = dependencies.platform || process.platform
  const arch = dependencies.arch || process.arch
  let state = { status: 'idle', message: 'Python과 SR 패키지를 확인하세요.', log: '', ready: false, installPath: locations.root }
  let active = null, controller = null, selectedPython = null
  const publish = update => { state = { ...state, ...update }; options.onStatus?.(state); return state }
  const log = chunk => publish({ log: (state.log + chunk).slice(-16000) })
  const command = (exe, args, extra = {}) => execute(exe, args, { signal: controller?.signal, ...extra })

  async function probe (candidate, prefix = []) {
    if (/WindowsApps/i.test(candidate)) return null
    try {
      const stdout = await command(candidate, [...prefix, '-I', '-c', PROBE], { timeout: 15000 })
      const line = stdout.split(/\r?\n/).find(line => line.startsWith(PROBE_PREFIX))
      const result = JSON.parse(line.slice(PROBE_PREFIX.length))
      return result.compatible ? result.python : null
    } catch (error) {
      if (controller?.signal.aborted) throw error
      return null
    }
  }

  async function inspect (python) {
    const stdout = await command(python, ['-B', '-I', path.join(__dirname, 'sr-probe-bootstrap.py'), locations.srRoot], { timeout: 180000 })
    return JSON.parse(stdout.trim())
  }

  async function check () {
    publish({ status: 'checking', ready: false, message: 'Python, 패키지, IMDN 가중치와 ROI 추론을 확인합니다.' })
    selectedPython = null
    const candidates = [locations.python, process.env.AVION_SR_PYTHON,
      ...(!options.isPackaged ? [locations.projectPython] : [])].filter(Boolean).map(file => [file, []])
    // Probe executables directly: newer Python launchers can auto-install on invocation.
    if (platform === 'win32') {
      for (const version of ['313', '312', '311', '310']) {
        for (const parent of [process.env.LOCALAPPDATA && path.join(process.env.LOCALAPPDATA, 'Programs', 'Python'), process.env.ProgramFiles].filter(Boolean)) {
          candidates.push([path.join(parent, `Python${version}`, 'python.exe'), []])
        }
      }
    }
    // Avoid invoking Windows' Store alias, which can open UI instead of running Python.
    const pathPython = (process.env.PATH || '').split(path.delimiter)
      .filter(dir => !/WindowsApps/i.test(dir)).map(dir => path.join(dir, platform === 'win32' ? 'python.exe' : 'python3'))
      .find(file => fs.existsSync(file))
    if (pathPython) candidates.push([pathPython, []])
    let first = null
    for (const [candidate, prefix] of candidates) {
      const python = await probe(candidate, prefix)
      if (!python) continue
      try {
        const report = await inspect(python)
        first ||= report
        selectedPython = python
        return publish({ status: report.ready ? 'ready' : 'missing', ready: report.ready, report,
          message: report.ready ? `준비 완료 · Python ${report.version} · ${report.cuda ? `CUDA (${report.device})` : 'CPU'} · IMDN/ROI 검증 통과`
            : '확인 완료 · 미설치 또는 오류가 있는 항목의 설치 버튼을 누르세요.' })
      } catch (error) {
        if (controller?.signal.aborted) throw error
        log(`${candidate}: ${error.message}\n`)
      }
    }
    return publish({ status: 'missing', ready: false, report: first,
      message: first ? 'SR 패키지 또는 모델 검증에 실패했습니다. 진단 내용을 확인하세요.' : '사용 가능한 Python 환경이 없습니다. Python 실행 환경 설치를 누르세요.' })
  }

  async function install (component) {
    if (platform !== 'win32' || arch !== 'x64') throw new Error('자동 설치는 Windows x64에서 지원합니다. 이 앱의 Windows x64 버전을 사용하세요.')
    const previous = state.report
    if (controller.signal.aborted) throw new Error('설치가 취소되었습니다.')
    fs.mkdirSync(locations.root, { recursive: true })
    publish({ status: 'installing', message: '앱 전용 Python을 준비합니다.', progress: null })
    // Reuse the project's venv, but never pip-install into a system/custom Python.
    const projectVenv = !options.isPackaged && selectedPython && previous?.isVenv &&
      path.resolve(selectedPython).toLowerCase() === path.resolve(locations.projectPython).toLowerCase()
    let target = projectVenv ? selectedPython : locations.python
    const needsEnvironment = !await probe(target)
    const base = needsEnvironment ? await probe(locations.basePython) : true
    if (!base) {
      const archive = path.join(locations.root, 'python-download.zip')
      await download(archive, { signal: controller.signal,
        onProgress: progress => publish({ progress, message: `Python ${PYTHON_VERSION} 다운로드${progress == null ? '' : ` ${progress}%`}` }) })
      const destination = path.dirname(path.dirname(locations.basePython))
      const staging = path.join(locations.root, `python-extract-${randomUUID()}`)
      publish({ progress: null, message: '검증한 Python 패키지를 설치합니다.' })
      const powershell = path.join(process.env.SystemRoot || 'C:\\Windows', 'System32', 'WindowsPowerShell', 'v1.0', 'powershell.exe')
      // Paths are environment values, never interpolated as PowerShell code.
      await command(powershell, ['-NoProfile', '-NonInteractive', '-Command',
        "$ErrorActionPreference='Stop'; [Console]::OutputEncoding=[Text.UTF8Encoding]::new(); Add-Type -AssemblyName System.IO.Compression.FileSystem; [IO.Compression.ZipFile]::ExtractToDirectory($env:AVION_PY_ARCHIVE,$env:AVION_PY_DEST)"],
      { env: { AVION_PY_ARCHIVE: archive, AVION_PY_DEST: staging }, timeout: 180000, onOutput: log })
      if (fs.existsSync(destination)) fs.renameSync(destination, `${destination}.backup-${Date.now()}`)
      fs.renameSync(staging, destination)
      fs.unlinkSync(archive)
      if (!await probe(locations.basePython)) throw new Error('다운로드한 Python을 실행할 수 없습니다. Windows x64 및 시스템 런타임을 확인하세요.')
    }
    const venv = path.dirname(path.dirname(target))
    if (needsEnvironment) {
      // Preserve an unusable environment (e.g. copied from a different PC) for diagnosis.
      if (fs.existsSync(venv)) fs.renameSync(venv, `${venv}.backup-${Date.now()}`)
      await command(locations.basePython, ['-I', '-m', 'venv', '--without-pip', venv], { onOutput: log })
    }
    publish({ message: 'pip를 준비합니다.' })
    await command(target, ['-I', '-m', 'ensurepip', '--upgrade'], { onOutput: log })
    const pip = ['-I', '-m', 'pip', '--isolated', '--disable-pip-version-check']
    if (component !== 'python') {
      publish({ message: `${COMPONENTS[component]} 설치 중 · 처음 설치할 때는 몇 분 걸릴 수 있습니다.` })
      const index = component === 'torch' ? 'https://download.pytorch.org/whl/cpu' : 'https://pypi.org/simple'
      const repair = previous?.packages?.[component] === null ? ['--force-reinstall'] : []
      await command(target, [...pip, 'install', '--only-binary=:all:', '--index-url', index,
        ...repair, COMPONENTS[component]], { timeout: 1800000, onOutput: log })
    }
    publish({ message: '설치 결과와 GPU 사용 가능 여부를 다시 확인합니다.', progress: null })
    const checked = await check()
    if (component !== 'python' && !checked.report?.packages?.[component]) throw new Error(`${component} 설치 후 검증 실패. 아래 진단 내용을 확인하세요.`)
    if (component === 'python' && !checked.report) throw new Error('Python 설치 후 검증 실패. 설치 로그를 확인하세요.')
    return checked
  }

  function startOperation (kind, component) {
    if (active) return active
    if (options.isInUse?.()) return Promise.resolve({ error: '측정 또는 DOOM 수집을 종료한 뒤 환경을 확인·설치하세요.' })
    if (kind === 'install') {
      if (!Object.hasOwn(COMPONENTS, component)) return Promise.resolve({ error: '지원하지 않는 설치 항목입니다.' })
      if (state.status === 'idle') return Promise.resolve({ error: '먼저 설치 여부 확인을 눌러주세요.' })
    }
    controller = new AbortController()
    publish({ log: '' })
    active = (kind === 'install' ? install(component) : check()).catch(error => publish({
      status: controller.signal.aborted ? 'cancelled' : 'failed', ready: false, progress: null,
      message: controller.signal.aborted ? '설치를 취소했습니다. 다시 시도할 수 있습니다.' : error.message,
      log: (state.log + '\n' + error.message).slice(-16000)
    })).finally(() => { active = null; controller = null })
    return active
  }

  return { check: () => startOperation('check'), install: component => startOperation('install', component),
    cancel: () => { controller?.abort(); return state },
    getState: () => state, isBusy: () => Boolean(active),
    requirePython: () => {
      if (active || !state.ready || !selectedPython) throw new Error('먼저 SR 실행 환경의 설치 여부를 확인하고 미설치 항목을 설치하세요.')
      return selectedPython
    } }
}

function registerEnvironment ({ app, ipcMain, windows, isInUse }) {
  const manager = createEnvironmentManager({ isPackaged: app.isPackaged, userData: app.getPath('userData'),
    resourcesPath: process.resourcesPath, repoRoot: path.resolve(__dirname, '..'), isInUse,
    onStatus: state => { for (const win of windows) if (!win.isDestroyed()) win.webContents.send('sr-environment:status', state) } })
  ipcMain.handle('sr-environment:get-state', () => manager.getState())
  ipcMain.handle('sr-environment:check', () => manager.check())
  ipcMain.handle('sr-environment:install', (_event, component) => manager.install(component))
  ipcMain.handle('sr-environment:cancel', () => manager.cancel())
  app.on('before-quit', () => manager.cancel())
  return manager
}

module.exports = { createEnvironmentManager, registerEnvironment, pathsFor, run, verifyArchive, PYTHON_VERSION }
