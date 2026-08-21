const { app, BrowserWindow, dialog, ipcMain, shell } = require('electron')
const { spawn } = require('child_process')
const fs = require('fs')
const path = require('path')

let captureEnabled = false
let captureProcess = null
let videoExtractionProcess = null
let lastRecordingPath = null
let lastRecordingMode = null
let captureSessionCounter = 0
const windows = new Set()

function getCaptureHostCandidates () {
  const executable = 'AvionCaptureHost.exe'
  const unpackedPath = process.resourcesPath
    ? path.join(process.resourcesPath, 'native', executable)
    : null
  const bundledPath = path.join(__dirname, 'native', executable)

  return [
    unpackedPath,
    bundledPath,
    path.join(__dirname, 'native', 'capture-host', 'build', 'Release', executable),
    path.join(__dirname, 'native', 'capture-host', 'build', 'Debug', executable)
  ].filter(Boolean)
}

function findCaptureHost () {
  return getCaptureHostCandidates().find((candidate) => fs.existsSync(candidate))
}

function broadcastCaptureStatus (status) {
  for (const win of windows) {
    if (!win.isDestroyed()) {
      win.webContents.send('capture:status', status)
    }
  }
}

function broadcastVideoFramesStatus (status) {
  for (const win of windows) {
    if (!win.isDestroyed()) {
      win.webContents.send('video-frames:status', status)
    }
  }
}

function captureState (overrides = {}) {
  return {
    enabled: captureEnabled,
    pid: captureProcess?.pid ?? null,
    recordingPath: lastRecordingPath,
    recordingMode: lastRecordingMode,
    ...overrides
  }
}

function timestampForFileName (date = new Date()) {
  const pad = (value) => String(value).padStart(2, '0')
  return [
    date.getFullYear(),
    pad(date.getMonth() + 1),
    pad(date.getDate())
  ].join('') + '-' + [
    pad(date.getHours()),
    pad(date.getMinutes()),
    pad(date.getSeconds())
  ].join('')
}

function createRecordingPath (recordingMode) {
  const recordingDir = path.join(app.getPath('videos'), 'Avion Captures')
  fs.mkdirSync(recordingDir, { recursive: true })
  const timestamp = timestampForFileName()
  const sessionId = String(++captureSessionCounter).padStart(3, '0')
  const baseName = `avion-capture-${timestamp}-${sessionId}`

  for (let attempt = 0; attempt < 1000; ++attempt) {
    const suffix = attempt === 0 ? '' : `-${String(attempt + 1).padStart(3, '0')}`
    const uniqueBaseName = `${baseName}${suffix}`
    const candidate = recordingMode === 'png-1fps'
      ? path.join(recordingDir, `${uniqueBaseName}-1fps`)
      : path.join(recordingDir, `${uniqueBaseName}.bgra`)

    if (!fs.existsSync(candidate)) {
      return candidate
    }
  }

  const fallbackName = `avion-capture-${timestamp}-${sessionId}-${Date.now()}`
  return recordingMode === 'png-1fps'
    ? path.join(recordingDir, `${fallbackName}-1fps`)
    : path.join(recordingDir, `${fallbackName}.bgra`)
}

function sanitizePathSegment (value) {
  return value.replace(/[<>:"/\\|?*\x00-\x1f]/g, '_').trim() || 'video'
}

function parseFrameRate (value) {
  if (typeof value !== 'string' || value === 'N/A') {
    return null
  }

  const trimmed = value.trim()
  const fraction = trimmed.match(/^(\d+(?:\.\d+)?)\/(\d+(?:\.\d+)?)$/)
  if (fraction) {
    const numerator = Number(fraction[1])
    const denominator = Number(fraction[2])
    return numerator > 0 && denominator > 0 ? numerator / denominator : null
  }

  const fps = Number(trimmed)
  return Number.isFinite(fps) && fps > 0 ? fps : null
}

function frameRateExpression (value, fps) {
  if (typeof value === 'string') {
    const trimmed = value.trim()
    const fraction = trimmed.match(/^(\d+)\/(\d+)$/)
    if (fraction && Number(fraction[1]) > 0 && Number(fraction[2]) > 0) {
      return trimmed
    }
  }

  if (!Number.isFinite(fps) || fps <= 0) {
    return null
  }

  return fps.toFixed(6).replace(/0+$/, '').replace(/\.$/, '')
}

function parseFiniteNumber (value) {
  const parsed = Number(value)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : null
}

function parseFrameCount (value) {
  if (typeof value !== 'string' || !/^\d+$/.test(value)) {
    return null
  }

  const parsed = Number(value)
  return Number.isSafeInteger(parsed) && parsed > 0 ? parsed : null
}

function parseOptionalFrameCount (value) {
  if (typeof value === 'number') {
    return Number.isSafeInteger(value) && value >= 0 ? value : null
  }

  if (typeof value !== 'string' || !/^\d+$/.test(value)) {
    return null
  }

  const parsed = Number(value)
  return Number.isSafeInteger(parsed) && parsed >= 0 ? parsed : null
}

function parseCsvLine (line) {
  const values = []
  let value = ''
  let inQuotes = false

  for (let index = 0; index < line.length; ++index) {
    const character = line[index]
    if (character === '"') {
      if (inQuotes && line[index + 1] === '"') {
        value += '"'
        ++index
      } else {
        inQuotes = !inQuotes
      }
      continue
    }

    if (character === ',' && !inQuotes) {
      values.push(value)
      value = ''
      continue
    }

    value += character
  }

  values.push(value)
  return values
}

function csvEscape (value) {
  const text = String(value ?? '')
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text
}

function parseDurationFromFfmpegOutput (stderr) {
  const match = stderr.match(/Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)/i)
  if (!match) {
    return null
  }

  return (Number(match[1]) * 3600) + (Number(match[2]) * 60) + Number(match[3])
}

function parseVideoInfoFromFfmpegOutput (stderr) {
  const fpsMatch = stderr.match(/,\s*([0-9]+(?:\.[0-9]+)?)\s*fps\b/i) ||
    stderr.match(/,\s*([0-9]+(?:\.[0-9]+)?)\s*tbr\b/i)
  const sizeMatch = stderr.match(/,\s*(\d{2,5})x(\d{2,5})(?:\s|,|\[)/i)
  const fps = fpsMatch ? parseFiniteNumber(fpsMatch[1]) : null

  if (!fps) {
    return null
  }

  const durationSeconds = parseDurationFromFfmpegOutput(stderr)
  return {
    fps,
    fpsExpression: frameRateExpression(null, fps),
    durationSeconds,
    estimatedFrameCount: durationSeconds ? Math.round(durationSeconds * fps) : null,
    width: sizeMatch ? Number(sizeMatch[1]) : null,
    height: sizeMatch ? Number(sizeMatch[2]) : null
  }
}

function runBufferedCommand (command, args) {
  return new Promise((resolve, reject) => {
    let child
    try {
      child = spawn(command, args, {
        windowsHide: true,
        stdio: ['ignore', 'pipe', 'pipe']
      })
    } catch (error) {
      reject(error)
      return
    }

    let stdout = ''
    let stderr = ''

    child.stdout?.on('data', (chunk) => {
      stdout += chunk.toString()
    })

    child.stderr?.on('data', (chunk) => {
      stderr += chunk.toString()
    })

    child.once('error', reject)
    child.once('close', (code) => {
      resolve({ code, stdout, stderr })
    })
  })
}

function commandErrorMessage (command, error) {
  if (error?.code === 'ENOENT') {
    return `${command} was not found on PATH. Install FFmpeg and restart Avion.`
  }

  return error?.message ?? String(error)
}

async function readVideoMetadataWithFfprobe (videoPath) {
  const result = await runBufferedCommand('ffprobe', [
    '-v',
    'error',
    '-select_streams',
    'v:0',
    '-show_entries',
    'stream=avg_frame_rate,r_frame_rate,nb_frames,duration,width,height',
    '-show_entries',
    'format=duration',
    '-of',
    'json',
    videoPath
  ])

  if (result.code !== 0) {
    throw new Error(result.stderr.trim() || result.stdout.trim() || `ffprobe exited with code ${result.code}`)
  }

  const metadata = JSON.parse(result.stdout)
  const stream = metadata.streams?.[0]
  if (!stream) {
    throw new Error('No video stream was found.')
  }

  const rateValue = parseFrameRate(stream.avg_frame_rate) ? stream.avg_frame_rate : stream.r_frame_rate
  const fps = parseFrameRate(rateValue)
  const durationSeconds = parseFiniteNumber(stream.duration) ?? parseFiniteNumber(metadata.format?.duration)
  const frameCount = parseFrameCount(stream.nb_frames)

  if (!fps) {
    throw new Error('Video FPS could not be read from ffprobe metadata.')
  }

  return {
    fps,
    fpsExpression: frameRateExpression(rateValue, fps),
    durationSeconds,
    estimatedFrameCount: frameCount ?? (durationSeconds ? Math.round(durationSeconds * fps) : null),
    width: Number(stream.width) || null,
    height: Number(stream.height) || null
  }
}

async function readVideoMetadata (videoPath) {
  let ffprobeFailure = null

  try {
    return await readVideoMetadataWithFfprobe(videoPath)
  } catch (error) {
    ffprobeFailure = commandErrorMessage('ffprobe', error)
  }

  try {
    const result = await runBufferedCommand('ffmpeg', ['-hide_banner', '-i', videoPath])
    const metadata = parseVideoInfoFromFfmpegOutput(result.stderr)
    if (metadata) {
      return metadata
    }
  } catch (error) {
    throw new Error(`Unable to analyze video FPS. ffprobe failed: ${ffprobeFailure}. ffmpeg failed: ${commandErrorMessage('ffmpeg', error)}`)
  }

  throw new Error(`Unable to analyze video FPS. ffprobe failed: ${ffprobeFailure}`)
}

function readJsonFile (filePath) {
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf8'))
  } catch (error) {
    throw new Error(`Unable to read JSON metadata: ${filePath}. ${error.message}`)
  }
}

function resolveSidecarPath (directory, value, fallback) {
  if (typeof value !== 'string' || value.length === 0) {
    return fallback
  }

  return path.isAbsolute(value) ? value : path.join(directory, value)
}

function resolveAvionCapturePaths (sourcePath) {
  const extension = path.extname(sourcePath).toLowerCase()
  if (!['.bgra', '.csv', '.json'].includes(extension)) {
    return null
  }

  const basePath = sourcePath.slice(0, sourcePath.length - extension.length)
  let dataPath = extension === '.bgra' ? sourcePath : `${basePath}.bgra`
  let timingPath = extension === '.csv' ? sourcePath : `${basePath}.csv`
  let metadataPath = extension === '.json' ? sourcePath : `${basePath}.json`

  if (!fs.existsSync(metadataPath)) {
    throw new Error(`Avion BGRA metadata was not found: ${metadataPath}`)
  }

  const metadata = readJsonFile(metadataPath)
  const metadataDir = path.dirname(metadataPath)
  dataPath = resolveSidecarPath(metadataDir, metadata.dataFile, dataPath)
  timingPath = resolveSidecarPath(metadataDir, metadata.timingFile, timingPath)

  if (!fs.existsSync(dataPath)) {
    throw new Error(`Avion BGRA data file was not found: ${dataPath}`)
  }

  if (!fs.existsSync(timingPath)) {
    throw new Error(`Avion frame timing CSV was not found: ${timingPath}`)
  }

  return {
    dataPath,
    timingPath,
    metadataPath,
    metadata
  }
}

function readAvionFrameTiming (timingPath) {
  const content = fs.readFileSync(timingPath, 'utf8').replace(/^\uFEFF/, '')
  const lines = content.split(/\r?\n/).filter((line) => line.trim().length > 0)
  if (lines.length === 0) {
    throw new Error(`Avion frame timing CSV is empty: ${timingPath}`)
  }

  const headers = parseCsvLine(lines[0])
  const indexOf = (name) => headers.indexOf(name)
  const frameIndexColumn = indexOf('frame_index')
  const timestamp100nsColumn = indexOf('timestamp_100ns')
  const timestampSecondsColumn = indexOf('timestamp_seconds')
  const delta100nsColumn = indexOf('delta_100ns')
  const instantFpsColumn = indexOf('instant_fps')
  const savedColumn = indexOf('saved')

  if (frameIndexColumn === -1 || timestamp100nsColumn === -1 || timestampSecondsColumn === -1 || savedColumn === -1) {
    throw new Error('Avion frame timing CSV is missing required columns.')
  }

  const savedRows = []
  let observedFrameCount = 0
  let lastTimestampSeconds = null

  for (let lineIndex = 1; lineIndex < lines.length; ++lineIndex) {
    const columns = parseCsvLine(lines[lineIndex])
    if (columns.length < headers.length) {
      continue
    }

    ++observedFrameCount
    const timestampSeconds = Number(columns[timestampSecondsColumn])
    if (Number.isFinite(timestampSeconds)) {
      lastTimestampSeconds = timestampSeconds
    }

    if (columns[savedColumn] !== '1') {
      continue
    }

    savedRows.push({
      frameIndex: columns[frameIndexColumn],
      timestamp100ns: columns[timestamp100nsColumn],
      timestampSeconds: columns[timestampSecondsColumn],
      delta100ns: delta100nsColumn === -1 ? '' : columns[delta100nsColumn],
      instantFps: instantFpsColumn === -1 ? '' : columns[instantFpsColumn]
    })
  }

  return {
    observedFrameCount,
    savedFrameCount: savedRows.length,
    durationSeconds: lastTimestampSeconds,
    savedRows
  }
}

function readAvionCaptureMetadata (sourcePath) {
  const paths = resolveAvionCapturePaths(sourcePath)
  if (!paths) {
    return null
  }

  const width = parseOptionalFrameCount(paths.metadata.width)
  const height = parseOptionalFrameCount(paths.metadata.height)
  const bytesPerPixel = parseOptionalFrameCount(paths.metadata.bytesPerPixel) ?? 4
  if (!width || !height || bytesPerPixel !== 4) {
    throw new Error('Avion BGRA metadata must include a valid width, height, and 4-byte BGRA pixel format.')
  }

  const frameSizeBytes = parseOptionalFrameCount(paths.metadata.frameSizeBytes) ?? (width * height * bytesPerPixel)
  if (!frameSizeBytes) {
    throw new Error('Avion BGRA frame size could not be determined.')
  }

  const dataSizeBytes = fs.statSync(paths.dataPath).size
  if (dataSizeBytes % frameSizeBytes !== 0) {
    throw new Error('Avion BGRA data size does not align with the metadata frame size. The capture may be incomplete.')
  }

  const timing = readAvionFrameTiming(paths.timingPath)
  const rawFrameCount = dataSizeBytes / frameSizeBytes
  if (rawFrameCount !== timing.savedFrameCount) {
    throw new Error(`Avion BGRA frame count mismatch. Raw frames: ${rawFrameCount}, CSV saved frames: ${timing.savedFrameCount}.`)
  }

  const durationSeconds = parseFiniteNumber(paths.metadata.durationSeconds) ?? timing.durationSeconds
  const averageFps = durationSeconds ? rawFrameCount / durationSeconds : parseFiniteNumber(paths.metadata.averageObservedFps)

  return {
    sourceType: 'avion-bgra',
    variableFrameRate: true,
    dataPath: paths.dataPath,
    timingPath: paths.timingPath,
    metadataPath: paths.metadataPath,
    width,
    height,
    frameSizeBytes,
    fps: averageFps,
    durationSeconds,
    estimatedFrameCount: rawFrameCount,
    observedFrameCount: timing.observedFrameCount,
    timingRows: timing.savedRows,
    originalMetadata: paths.metadata
  }
}

function createVideoFramesOutputDir (videoPath) {
  const outputRoot = path.join(app.getPath('pictures'), 'Avion Video Frames')
  const sourceName = sanitizePathSegment(path.parse(videoPath).name)
  const baseName = `${sourceName}-${timestampForFileName()}`
  fs.mkdirSync(outputRoot, { recursive: true })

  for (let attempt = 0; attempt < 1000; ++attempt) {
    const suffix = attempt === 0 ? '' : `-${String(attempt + 1).padStart(3, '0')}`
    const candidate = path.join(outputRoot, `${baseName}${suffix}`)
    if (!fs.existsSync(candidate)) {
      return candidate
    }
  }

  return path.join(outputRoot, `${baseName}-${Date.now()}`)
}

function countPngFrames (outputDir) {
  try {
    return fs.readdirSync(outputDir).filter((name) => name.toLowerCase().endsWith('.png')).length
  } catch {
    return 0
  }
}

function parseFfmpegProgressTime (key, value) {
  if (key === 'out_time_us' || key === 'out_time_ms') {
    const parsed = Number(value)
    return Number.isFinite(parsed) ? parsed / 1000000 : null
  }

  if (key === 'out_time') {
    const match = value.match(/^(\d+):(\d+):(\d+(?:\.\d+)?)$/)
    if (match) {
      return (Number(match[1]) * 3600) + (Number(match[2]) * 60) + Number(match[3])
    }
  }

  return null
}

function writeAvionOutputMetadata (outputDir, capture, frameCount) {
  const metadata = {
    format: 'avion.frames.png-sequence',
    sourceType: capture.sourceType,
    sourceDataFile: capture.dataPath,
    sourceTimingFile: capture.timingPath,
    sourceMetadataFile: capture.metadataPath,
    frameDirectory: path.basename(outputDir),
    timingFile: 'frames.csv',
    width: capture.width,
    height: capture.height,
    pixelFormat: 'png',
    sourcePixelFormat: 'bgra',
    variableFrameRate: true,
    durationSeconds: capture.durationSeconds,
    averageObservedFps: capture.fps,
    frameCount,
    losslessSource: true
  }

  fs.writeFileSync(path.join(outputDir, 'capture.json'), `${JSON.stringify(metadata, null, 2)}\n`)
}

function writeAvionFrameManifest (outputDir, capture, frameCount) {
  const lines = ['frame_index,timestamp_100ns,timestamp_seconds,delta_100ns,instant_fps,saved,output_file']
  const rowsToWrite = Math.min(frameCount, capture.timingRows.length)

  for (let index = 0; index < rowsToWrite; ++index) {
    const row = capture.timingRows[index]
    const outputFile = `frame_${String(index + 1).padStart(6, '0')}.png`
    lines.push([
      row.frameIndex,
      row.timestamp100ns,
      row.timestampSeconds,
      row.delta100ns,
      row.instantFps,
      '1',
      outputFile
    ].map(csvEscape).join(','))
  }

  fs.writeFileSync(path.join(outputDir, 'frames.csv'), `${lines.join('\n')}\n`)
}

async function extractAvionCaptureFrames (sourcePath) {
  if (videoExtractionProcess && !videoExtractionProcess.killed) {
    throw new Error('A frame extraction is already running.')
  }

  broadcastVideoFramesStatus({
    status: 'analyzing',
    videoPath: sourcePath
  })

  const capture = readAvionCaptureMetadata(sourcePath)
  const outputDir = createVideoFramesOutputDir(capture.dataPath)
  fs.mkdirSync(outputDir, { recursive: true })

  const metadata = {
    sourceType: capture.sourceType,
    variableFrameRate: true,
    fps: capture.fps,
    durationSeconds: capture.durationSeconds,
    estimatedFrameCount: capture.estimatedFrameCount,
    width: capture.width,
    height: capture.height
  }
  const outputPattern = path.join(outputDir, 'frame_%06d.png')

  return new Promise((resolve, reject) => {
    const args = [
      '-hide_banner',
      '-y',
      '-f',
      'rawvideo',
      '-pixel_format',
      'bgra',
      '-video_size',
      `${capture.width}x${capture.height}`,
      '-i',
      capture.dataPath,
      '-vsync',
      '0',
      '-start_number',
      '1',
      '-progress',
      'pipe:1',
      '-nostats',
      outputPattern
    ]

    let child
    try {
      child = spawn('ffmpeg', args, {
        windowsHide: true,
        stdio: ['ignore', 'pipe', 'pipe']
      })
    } catch (error) {
      reject(new Error(commandErrorMessage('ffmpeg', error)))
      return
    }

    videoExtractionProcess = child
    let stderr = ''
    let progressBuffer = ''
    let currentFrame = 0

    const sendProgress = () => {
      const percent = capture.estimatedFrameCount
        ? Math.max(0, Math.min(99, Math.round((currentFrame / capture.estimatedFrameCount) * 100)))
        : null

      broadcastVideoFramesStatus({
        status: 'extracting',
        videoPath: sourcePath,
        outputDir,
        metadata,
        frame: currentFrame,
        percent
      })
    }

    broadcastVideoFramesStatus({
      status: 'extracting',
      videoPath: sourcePath,
      outputDir,
      metadata,
      frame: 0,
      percent: 0
    })

    child.stdout?.on('data', (chunk) => {
      progressBuffer += chunk.toString()
      const lines = progressBuffer.split(/\r?\n/)
      progressBuffer = lines.pop() ?? ''

      for (const line of lines) {
        const separator = line.indexOf('=')
        if (separator === -1 || line.slice(0, separator) !== 'frame') {
          continue
        }

        const parsed = Number(line.slice(separator + 1))
        if (Number.isFinite(parsed)) {
          currentFrame = parsed
          sendProgress()
        }
      }
    })

    child.stderr?.on('data', (chunk) => {
      stderr += chunk.toString()
    })

    child.once('error', (error) => {
      if (videoExtractionProcess === child) {
        videoExtractionProcess = null
      }
      reject(new Error(commandErrorMessage('ffmpeg', error)))
    })

    child.once('close', (code) => {
      if (videoExtractionProcess === child) {
        videoExtractionProcess = null
      }

      const frameCount = countPngFrames(outputDir)
      if (code !== 0) {
        const detail = stderr.trim().split(/\r?\n/).slice(-6).join('\n')
        reject(new Error(detail || `ffmpeg exited with code ${code}`))
        return
      }

      if (frameCount !== capture.estimatedFrameCount) {
        reject(new Error(`Extracted PNG count mismatch. PNG frames: ${frameCount}, BGRA frames: ${capture.estimatedFrameCount}.`))
        return
      }

      writeAvionFrameManifest(outputDir, capture, frameCount)
      writeAvionOutputMetadata(outputDir, capture, frameCount)

      const result = {
        status: 'completed',
        videoPath: sourcePath,
        outputDir,
        metadata,
        frameCount,
        percent: 100
      }
      broadcastVideoFramesStatus(result)
      resolve(result)
    })
  })
}

async function extractVideoFrames (videoPath) {
  if (videoExtractionProcess && !videoExtractionProcess.killed) {
    throw new Error('A video frame extraction is already running.')
  }

  if (!fs.existsSync(videoPath)) {
    throw new Error(`Video file was not found: ${videoPath}`)
  }

  broadcastVideoFramesStatus({
    status: 'analyzing',
    videoPath
  })

  const metadata = await readVideoMetadata(videoPath)
  if (!metadata.fpsExpression) {
    throw new Error('Video FPS could not be converted into an ffmpeg fps filter.')
  }

  const outputDir = createVideoFramesOutputDir(videoPath)
  fs.mkdirSync(outputDir, { recursive: true })

  const outputPattern = path.join(outputDir, 'frame_%06d.png')
  return new Promise((resolve, reject) => {
    const args = [
      '-hide_banner',
      '-y',
      '-i',
      videoPath,
      '-map',
      '0:v:0',
      '-vf',
      `fps=${metadata.fpsExpression}`,
      '-start_number',
      '1',
      '-progress',
      'pipe:1',
      '-nostats',
      outputPattern
    ]

    let child
    try {
      child = spawn('ffmpeg', args, {
        windowsHide: true,
        stdio: ['ignore', 'pipe', 'pipe']
      })
    } catch (error) {
      reject(error)
      return
    }

    videoExtractionProcess = child
    let stderr = ''
    let progressBuffer = ''
    let currentFrame = null
    let currentTime = null

    const sendProgress = () => {
      const percent = metadata.durationSeconds && currentTime !== null
        ? Math.max(0, Math.min(99, Math.round((currentTime / metadata.durationSeconds) * 100)))
        : null

      broadcastVideoFramesStatus({
        status: 'extracting',
        videoPath,
        outputDir,
        metadata,
        frame: currentFrame,
        percent
      })
    }

    broadcastVideoFramesStatus({
      status: 'extracting',
      videoPath,
      outputDir,
      metadata,
      frame: 0,
      percent: 0
    })

    child.stdout?.on('data', (chunk) => {
      progressBuffer += chunk.toString()
      const lines = progressBuffer.split(/\r?\n/)
      progressBuffer = lines.pop() ?? ''

      for (const line of lines) {
        const separator = line.indexOf('=')
        if (separator === -1) {
          continue
        }

        const key = line.slice(0, separator)
        const value = line.slice(separator + 1)
        if (key === 'frame') {
          const parsed = Number(value)
          currentFrame = Number.isFinite(parsed) ? parsed : currentFrame
          sendProgress()
          continue
        }

        const parsedTime = parseFfmpegProgressTime(key, value)
        if (parsedTime !== null) {
          currentTime = parsedTime
          sendProgress()
        }
      }
    })

    child.stderr?.on('data', (chunk) => {
      stderr += chunk.toString()
    })

    child.once('error', (error) => {
      if (videoExtractionProcess === child) {
        videoExtractionProcess = null
      }
      reject(new Error(commandErrorMessage('ffmpeg', error)))
    })

    child.once('close', (code) => {
      if (videoExtractionProcess === child) {
        videoExtractionProcess = null
      }

      const frameCount = countPngFrames(outputDir)
      if (code !== 0) {
        const detail = stderr.trim().split(/\r?\n/).slice(-6).join('\n')
        reject(new Error(detail || `ffmpeg exited with code ${code}`))
        return
      }

      const result = {
        status: 'completed',
        videoPath,
        outputDir,
        metadata,
        frameCount,
        percent: 100
      }
      broadcastVideoFramesStatus(result)
      resolve(result)
    })
  })
}

async function extractFramesFromSource (sourcePath) {
  if (resolveAvionCapturePaths(sourcePath)) {
    return extractAvionCaptureFrames(sourcePath)
  }

  return extractVideoFrames(sourcePath)
}

function startCaptureHost (options = {}) {
  if (captureProcess && !captureProcess.killed) {
    return captureState({ status: captureEnabled ? 'running' : 'stopping' })
  }

  const hostPath = findCaptureHost()
  if (!hostPath) {
    captureEnabled = false
    return captureState({
      status: 'missing-host',
      message: 'Native capture host is not built.'
    })
  }

  try {
    const recordingMode = options.oneFramePerSecond ? 'png-1fps' : 'raw-bgra'
    lastRecordingMode = recordingMode
    lastRecordingPath = createRecordingPath(recordingMode)
    captureProcess = spawn(hostPath, ['--record', '--record-mode', recordingMode, '--record-path', lastRecordingPath], {
      cwd: path.dirname(hostPath),
      detached: false,
      stdio: ['pipe', 'ignore', 'ignore'],
      windowsHide: false
    })
  } catch (error) {
    captureEnabled = false
    captureProcess = null
    lastRecordingPath = null
    lastRecordingMode = null
    return captureState({
      status: 'failed',
      message: error.message
    })
  }

  captureEnabled = true
  const child = captureProcess

  child.once('error', (error) => {
    if (captureProcess !== child) {
      return
    }

    captureEnabled = false
    captureProcess = null
    broadcastCaptureStatus(captureState({
      status: 'failed',
      message: error.message
    }))
  })

  child.once('exit', (code, signal) => {
    if (captureProcess !== child) {
      return
    }

    captureEnabled = false
    captureProcess = null
    broadcastCaptureStatus(captureState({
      status: code === 0 || signal === 'SIGTERM' ? 'stopped' : 'exited',
      exitCode: code,
      signal
    }))
  })

  child.unref()
  return captureState({ status: 'running', recordingMode: options.oneFramePerSecond ? 'png-1fps' : 'raw-bgra' })
}

function stopCaptureHost () {
  if (captureProcess && !captureProcess.killed) {
    const child = captureProcess
    captureEnabled = false

    try {
      child.stdin?.write('stop\n')
      child.stdin?.end()
    } catch {
      child.kill()
    }

    setTimeout(() => {
      if (captureProcess === child && !child.killed) {
        child.kill()
      }
    }, 3000)

    return captureState({ status: 'stopping' })
  }

  captureProcess = null
  captureEnabled = false
  return captureState({ status: 'stopped' })
}

function createWindow () {
  const win = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js')
    }
  })
  windows.add(win)
  win.on('closed', () => windows.delete(win))
  win.loadFile('index.html')
}

ipcMain.handle('capture:set-enabled', (_event, enabled, options = {}) => {
  const state = enabled ? startCaptureHost(options) : stopCaptureHost()
  broadcastCaptureStatus(state)
  return state
})

ipcMain.handle('capture:get-status', () => {
  return captureState({ status: captureEnabled ? 'running' : 'stopped' })
})

ipcMain.handle('video-frames:select-and-extract', async (event) => {
  const win = BrowserWindow.fromWebContents(event.sender)
  const dialogOptions = {
    title: 'Select video or Avion BGRA capture',
    properties: ['openFile'],
    filters: [
      {
        name: 'Videos and Avion Captures',
        extensions: ['mp4', 'mov', 'mkv', 'avi', 'webm', 'm4v', 'wmv', 'bgra', 'csv', 'json']
      },
      {
        name: 'Avion BGRA Captures',
        extensions: ['bgra', 'csv', 'json']
      },
      {
        name: 'Videos',
        extensions: ['mp4', 'mov', 'mkv', 'avi', 'webm', 'm4v', 'wmv']
      },
      {
        name: 'All Files',
        extensions: ['*']
      }
    ]
  }
  const result = win
    ? await dialog.showOpenDialog(win, dialogOptions)
    : await dialog.showOpenDialog(dialogOptions)

  if (result.canceled || result.filePaths.length === 0) {
    return { status: 'cancelled' }
  }

  try {
    return await extractFramesFromSource(result.filePaths[0])
  } catch (error) {
    const failedState = {
      status: 'failed',
      message: error.message
    }
    broadcastVideoFramesStatus(failedState)
    return failedState
  }
})

ipcMain.handle('video-frames:open-output', async (_event, outputDir) => {
  if (!outputDir || !fs.existsSync(outputDir)) {
    return {
      success: false,
      message: 'Output folder was not found.'
    }
  }

  const message = await shell.openPath(outputDir)
  return {
    success: message.length === 0,
    message
  }
})

app.whenReady().then(() => {
  createWindow()
})

app.on('before-quit', () => {
  stopCaptureHost()
})

app.on('window-all-closed', function () {
  if (process.platform !== 'darwin') app.quit()
})
