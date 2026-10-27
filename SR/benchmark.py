"""Reproducible, full-reference SR evaluation. stdout is a JSON-lines IPC stream.

Timing includes RGB numpy -> model tensor -> CPU RGB uint8, excluding file I/O,
quality metrics, warmup and model loading. It is NOT capture-to-display latency.
"""
import argparse
import base64
import csv
import copy
import hashlib
import html
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

sys.dont_write_bytecode = True
import cv2
import numpy as np
from skimage.metrics import structural_similarity
from roi_dataset import MANIFEST, read_frames, remap_regions, region_mask
from sr_worker import selective_upscale, prepare_regions, FRAME_FLAG_BUDGET, FRAME_FLAG_BACKGROUND

ROOT = Path(__file__).resolve().parent
METHODS = ('nearest', 'bilinear', 'bicubic', 'imdn', 'imdn-fp16', 'imdn-roi', 'imdn-roi-fp16')
LABELS = {'nearest': 'Nearest', 'bilinear': 'Bilinear', 'bicubic': 'Bicubic',
          'imdn': '전체 SR FP32', 'imdn-fp16': '전체 SR FP16',
          'imdn-roi': 'ROI SR FP32', 'imdn-roi-fp16': 'ROI SR FP16'}
HIDDEN = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}


def emit(**event):
    print(json.dumps(event, ensure_ascii=False, allow_nan=False), flush=True)


def image_read(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'이미지를 읽을 수 없습니다: {path}')
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def image_write(path, rgb):
    ok, encoded = cv2.imencode('.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError('PNG 인코딩 실패')
    encoded.tofile(path)


def prepare_pair(reference, scale, max_edge=512):
    """Synthetic LR derived from the reference; never upsample LR to invent GT."""
    height, width = reference.shape[:2]
    if max_edge and max(height, width) > max_edge:
        ratio = max_edge / max(height, width)
        reference = cv2.resize(reference, (max(1, round(width * ratio)),
                                          max(1, round(height * ratio))),
                               interpolation=cv2.INTER_AREA)
    height, width = reference.shape[:2]
    height, width = height // scale * scale, width // scale * scale
    if min(height, width) < 11 + 2 * scale:
        raise ValueError('기준 영상이 너무 작습니다. 테두리 제외 후 최소 11×11이 필요합니다.')
    reference = np.ascontiguousarray(reference[:height, :width])
    low = cv2.resize(reference, (width // scale, height // scale),
                     interpolation=cv2.INTER_CUBIC)
    return reference, low


def quality(reference, output, border):
    if reference.shape != output.shape:
        raise ValueError(f'기준/출력 크기 불일치: {reference.shape} / {output.shape}')
    if border:
        reference = reference[border:-border, border:-border]
        output = output[border:-border, border:-border]
    error = reference.astype(np.float64) - output.astype(np.float64)
    mse = float(np.mean(error ** 2))
    # JSON has no Infinity. Preserve exact-match semantics explicitly.
    psnr = None if mse == 0 else 10 * math.log10(255 ** 2 / mse)
    ssim = structural_similarity(reference, output, data_range=255, channel_axis=-1,
                                 gaussian_weights=True, sigma=1.5,
                                 use_sample_covariance=False)
    return {'psnr_db': psnr, 'psnr_exact': mse == 0, 'ssim': float(ssim)}


def latency_summary(samples):
    if not samples or min(samples) <= 0:
        raise ValueError('양수 지연 표본이 필요합니다.')
    values = np.asarray(samples, dtype=float)
    return {'latency_mean_ms': float(values.mean()),
            'latency_p50_ms': float(np.median(values)),
            'latency_p95_ms': float(np.percentile(values, 95)),
            'processing_fps': float(len(values) * 1000 / values.sum()),
            'timed_frames': len(samples)}


def roi_quality(reference, output, regions, border):
    mask = region_mask(regions, reference.shape[1], reference.shape[0])
    if border:
        mask = mask[border:-border, border:-border]
        reference = reference[border:-border, border:-border]
        output = output[border:-border, border:-border]
    result = {'roi_psnr_db': None, 'roi_psnr_exact': False, 'roi_ssim': None,
              'roi_pixels': int(mask.sum()), 'roi_ssim_pixels': 0}
    if not mask.any():
        return result
    mse = float(np.mean((reference.astype(float)[mask] - output.astype(float)[mask]) ** 2))
    result.update(roi_psnr_db=None if mse == 0 else 10 * math.log10(255 ** 2 / mse), roi_psnr_exact=mse == 0)
    # Only score SSIM windows fully inside the union of object rectangles.
    valid = cv2.erode(mask.astype(np.uint8), np.ones((11, 11), np.uint8),
                      borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    if valid.any():
        _, ssim_map = structural_similarity(reference, output, data_range=255, channel_axis=-1,
                                            gaussian_weights=True, sigma=1.5,
                                            use_sample_covariance=False, full=True)
        result.update(roi_ssim=float(ssim_map[valid].mean()), roi_ssim_pixels=int(valid.sum()))
    return result


def average_quality(rows):
    result = {}
    for prefix in ('', 'roi_'):
        exact = any(row.get(prefix + 'psnr_exact', False) for row in rows)
        values = [row[prefix + 'psnr_db'] for row in rows if row.get(prefix + 'psnr_db') is not None]
        ssims = [row[prefix + 'ssim'] for row in rows if row.get(prefix + 'ssim') is not None]
        result.update({prefix + 'psnr_exact': exact,
                       prefix + 'psnr_db': None if exact or not values else float(np.mean(values)),
                       prefix + 'ssim': float(np.mean(ssims)) if ssims else None})
    result['roi_metric_frames'] = sum(row.get('roi_pixels', 0) > 0 for row in rows)
    result['roi_ssim_frames'] = sum(row.get('roi_ssim') is not None for row in rows)
    result['all_frames_exact'] = bool(rows) and all(row.get('psnr_exact', False) for row in rows)
    return result


def capture_frames(source, count):
    metadata_path = source.with_suffix('.json')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8-sig'))
    if metadata.get('format') != 'avion.lossless.bgra' or metadata.get('pixelFormat') != 'bgra':
        raise ValueError('Avion BGRA 캡처 메타데이터가 아닙니다.')
    if metadata.get('rowOrder') != 'top-down' or metadata.get('bytesPerPixel') != 4:
        raise ValueError('지원하지 않는 캡처 픽셀 배치입니다.')
    width, height = int(metadata['width']), int(metadata['height'])
    stride = int(metadata['frameStrideBytes'])
    frame_bytes = int(metadata['frameSizeBytes'])
    if not (0 < width <= 8192 and 0 < height <= 8192 and stride >= width * 4
            and frame_bytes == stride * height and frame_bytes <= 256 * 1024 ** 2):
        raise ValueError('캡처 크기/스트라이드가 올바르지 않습니다.')
    data_file = metadata_path.parent / metadata['dataFile']
    if data_file.resolve().parent != metadata_path.resolve().parent:
        raise ValueError('캡처 데이터는 메타데이터와 같은 폴더에 있어야 합니다.')
    size = data_file.stat().st_size
    saved = int(metadata['savedFrameCount'])
    if size % frame_bytes or size // frame_bytes != saved or saved <= 0:
        raise ValueError('캡처 파일 크기와 저장 프레임 수가 일치하지 않습니다.')
    indexes = np.linspace(0, saved - 1, min(count, saved), dtype=int)
    with data_file.open('rb') as stream:
        for index in indexes:
            stream.seek(int(index) * frame_bytes)
            raw = np.frombuffer(stream.read(frame_bytes), np.uint8).reshape(height, stride)
            rgb = raw[:, :width * 4].reshape(height, width, 4)[:, :, [2, 1, 0]].copy()
            yield f'frame-{index + 1:06}', rgb


def load_dataset(args):
    source = Path(args.source).resolve() if args.source else ROOT / 'IMDN/Test_Datasets/Set5'
    roi_source = source.name == MANIFEST or (source.is_dir() and (source / MANIFEST).is_file())
    if any('roi' in method for method in args.methods) and not roi_source:
        raise ValueError('ROI 비교에는 DOOM 장면 수집으로 저장한 roi-manifest.json이 필요합니다.')
    args.roi_data, args.frame_metadata = [], []
    if roi_source:
        images = read_frames(source, args.max_frames)
        kind = 'DOOM 동일 장면 · 전체 SR / ROI SR'
    elif source.is_file() and source.suffix.lower() in ('.json', '.bgra', '.csv'):
        images = ((name, rgb, []) for name, rgb in capture_frames(source, args.max_frames))
        kind = 'Avion 캡처 재구성 평가'
    elif source.is_dir():
        files = sorted(p for p in source.iterdir()
                       if p.suffix.lower() in ('.png', '.bmp', '.jpg', '.jpeg', '.webp'))
        if not files:
            raise ValueError('선택한 폴더에 이미지가 없습니다.')
        # Sample the full sequence rather than only its opening frames.
        selected = np.linspace(0, len(files) - 1, min(args.max_frames, len(files)), dtype=int)
        images = ((files[i].name, image_read(files[i]), []) for i in selected)
        kind = '사용자 프레임 재구성 평가' if args.source else 'Set5 샘플 재구성 평가'
    else:
        raise ValueError(f'이미지 폴더 또는 Avion 캡처를 선택하세요: {source}')
    pairs = []
    total_bytes = 0
    for name, rgb, regions in images:
        native = args.mode == 'native'
        if native:
            reference, low = rgb, rgb
            ref_regions = regions
        else:
            reference, low = prepare_pair(rgb, args.scale, args.max_edge)
            h, w = rgb.shape[:2]
            ratio = min(1, args.max_edge / max(h, w)) if args.max_edge else 1
            ref_regions = remap_regions(regions, round(w * ratio) / w, round(h * ratio) / h,
                                        reference.shape[1], reference.shape[0])
        lr_regions = ref_regions if native else remap_regions(ref_regions, 1 / args.scale, 1 / args.scale, low.shape[1], low.shape[0])
        args.roi_data.append((ref_regions, lr_regions))
        boxes = prepare_regions(lr_regions, low.shape[1], low.shape[0])
        args.frame_metadata.append({'name': name, 'input_size': list(low.shape[:2][::-1]),
            'output_size': [low.shape[1] * args.scale, low.shape[0] * args.scale],
            'reference_sha256': None if native else hashlib.sha256(reference.tobytes()).hexdigest(),
            'input_sha256': hashlib.sha256(low.tobytes()).hexdigest(),
            'source_rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(),
            'source_regions': regions, 'input_regions': lr_regions,
            'roi_coverage_pct': float(region_mask(lr_regions, low.shape[1], low.shape[0]).mean() * 100),
            'padded_coverage_pct': sum((r-l)*(b-t) for l,t,r,b in boxes) / (low.shape[1] * low.shape[0]) * 100})
        total_bytes += reference.nbytes + low.nbytes
        if total_bytes > 512 * 1024 ** 2:
            raise ValueError('평가 데이터가 512 MiB를 초과했습니다. 프레임 수나 최대 크기를 줄이세요.')
        pairs.append((name, reference, low))
    if not pairs:
        raise ValueError('평가할 프레임이 없습니다.')
    return source, kind + (' · 원본 입력 속도 평가 (GT 없음)' if args.mode == 'native' else ' · 축소/복원 화질 평가'), pairs


class GpuSampler:
    """Persistent nvidia-smi subprocess; unsupported/missing readings stay null."""
    def __init__(self, gpu_id=None):
        self.process = None
        self.thread = None
        self.samples = []
        self.gpu_id = gpu_id
        self.reason = 'CUDA 평가에서만 GPU를 계측합니다.' if not gpu_id else None

    def start(self):
        if not self.gpu_id:
            return
        try:
            self.process = subprocess.Popen([
                'nvidia-smi', '-i', self.gpu_id,
                '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits',
                '--loop-ms=100'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, encoding='utf-8', **HIDDEN)
            self.thread = threading.Thread(target=self._read, daemon=True)
            self.thread.start()
        except OSError as exc:
            self.reason = f'nvidia-smi 사용 불가: {exc}'

    def _read(self):
        for line in self.process.stdout:
            try:
                usage, memory = [float(x.strip()) for x in line.strip().split(',')]
                if 0 <= usage <= 100 and memory >= 0:
                    self.samples.append((time.perf_counter(), usage, memory))
            except (ValueError, TypeError):
                continue

    def summarize(self, start, end):
        samples = [(t, u, m) for t, u, m in self.samples if start <= t <= end]
        return {'gpu_utilization_mean_pct': float(np.mean([s[1] for s in samples])) if samples else None,
                'gpu_memory_peak_mib': max(s[2] for s in samples) if samples else None,
                'gpu_samples': len(samples),
                'gpu_unavailable_reason': None if samples else (self.reason or '측정 구간에 유효한 GPU 표본이 없습니다.')}

    def stop(self):
        if self.process:
            if self.process.poll() is None:
                self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
            if self.thread:
                self.thread.join(timeout=2)
            self.process.stdout.close()


def load_model(args):
    import torch
    from IMDN.model.architecture import IMDN
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA를 사용할 수 없습니다. CPU 또는 자동 선택을 사용하세요.')
    if any(m.endswith('fp16') for m in args.methods) and device.type != 'cuda':
        raise ValueError('IMDN FP16 비교에는 CUDA GPU가 필요합니다.')
    checkpoint = Path(args.checkpoint) if args.checkpoint else ROOT / f'IMDN/checkpoints/IMDN_x{args.scale}.pth'
    if not checkpoint.is_file():
        raise ValueError(f'IMDN 가중치가 없습니다: {checkpoint}')
    weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
    weights = {key.removeprefix('module.'): value for key, value in weights.items()}
    model = IMDN(upscale=args.scale)
    model.load_state_dict(weights, strict=True)
    model.eval().to(device)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = False
    return torch, model, device, checkpoint


def model_runner(torch, model, device, half=False):
    dtype = torch.float16 if half else torch.float32
    # Preserve FP32 checkpoint weights even when callers request FP16 first.
    runtime_model = copy.deepcopy(model) if half else model
    runtime_model.to(dtype=dtype)

    def run(rgb, validate=False):
        # Same quantization and transfer path as the live sr_worker.model_upscale.
        tensor = torch.from_numpy(rgb).to(device=device)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(dtype=dtype).div_(255.0)
        with torch.inference_mode():
            output = runtime_model(tensor)
            if validate and not torch.isfinite(output).all().item():
                raise ValueError('모델 출력에 NaN/Infinity가 있습니다. 해당 결과를 성능 수치로 저장하지 않습니다.')
            output = output.clamp_(0.0, 1.0)
            output = output.squeeze(0).permute(1, 2, 0).mul_(255.0).byte()
            return output.cpu().numpy()
    return run


def display(value, digits=2):
    return 'N/A' if value is None else f'{value:.{digits}f}'


def write_report(output, result):
    esc = lambda value: html.escape(str(value))
    rows = ''.join('<tr>' + ''.join(f'<td>{esc(v)}</td>' for v in [
        row['label'], row['device'], '∞' if row['psnr_exact'] else display(row['psnr_db']),
        display(row['ssim'], 4), '∞' if row['roi_psnr_exact'] else display(row['roi_psnr_db']),
        display(row['roi_ssim'], 4), display(row['latency_mean_ms']), display(row['latency_p95_ms']),
        display(row['processing_fps']), display(row['gpu_utilization_mean_pct']), row['execution_summary']])
        + '</tr>' for row in result['results'])
    charts = ''
    for metric, title in [('psnr_db', 'PSNR (dB, 높을수록 좋음)'),
                          ('latency_mean_ms', '프레임 처리 지연 (ms, 낮을수록 좋음)'),
                          ('processing_fps', '처리량 (FPS)'), ('gpu_utilization_mean_pct', 'GPU 전체 사용률 (%)')]:
        maximum = max((r[metric] or 0 for r in result['results']), default=1) or 1
        bars = ''.join(f'<div class="bar-row"><span>{esc(r["label"])}</span><div class="track"><i style="width:{(r[metric] or 0)/maximum*100:.2f}%"></i></div><b>{display(r[metric])}</b></div>' for r in result['results'])
        charts += f'<section><h2>{title}</h2>{bars}</section>'
    previews = ''
    for preview in result['previews']:
        data = base64.b64encode((output / preview['file']).read_bytes()).decode('ascii')
        previews += f'<figure><img src="data:image/png;base64,{data}" alt="{esc(preview["label"])}"><figcaption>{esc(preview["label"])}</figcaption></figure>'
    notes = ''.join(f'<li>{esc(note)}</li>' for note in result['protocol']['notes'])
    document = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>AVION 성능평가</title>
<style>body{{font-family:Segoe UI,Malgun Gothic,sans-serif;margin:40px auto;padding:0 24px;max-width:1200px;color:#172438;background:#f5f7fb}}h1{{font-size:36px}}h2{{font-size:19px}}table{{width:100%;border-collapse:collapse;background:white}}th,td{{padding:12px;text-align:right;border-bottom:1px solid #d5dce8}}th:first-child,td:first-child{{text-align:left}}.charts{{display:grid;grid-template-columns:1fr 1fr;gap:30px}}.bar-row{{display:flex;gap:10px;align-items:center;margin:12px 0}}.bar-row span{{width:100px}}.bar-row b{{width:75px}}.track{{flex:1;background:#dfe8f5;height:18px}}i{{display:block;height:100%;background:#2856a3}}.previews{{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}}figure{{margin:0}}img{{width:100%;image-rendering:pixelated}}figcaption{{margin:8px 0 20px}}li{{margin:8px 0;line-height:1.5}}.scroll{{overflow:auto}}@media(max-width:700px){{.charts,.previews{{grid-template-columns:1fr}}}}@media print{{body{{margin:15px;background:white}}}}</style>
<h1>AVION 성능평가</h1><p>{esc(result['dataset']['label'])} / ×{result['protocol']['scale']} / {result['dataset']['frames']}개 기준 프레임 / {esc(result['environment']['device_name'])}</p>
<p>측정 시각: {esc(result['created_at'])} · 실행 ID: {esc(result['run_id'])}</p>
<div class="scroll"><table><thead><tr><th>방식</th><th>장치</th><th>전체 PSNR dB</th><th>전체 SSIM</th><th>ROI PSNR dB</th><th>ROI SSIM</th><th>평균 ms</th><th>P95 ms</th><th>처리 FPS</th><th>GPU %</th><th>실제 처리 / 장면 수</th></tr></thead><tbody>{rows}</tbody></table></div>
<div class="charts">{charts}</div><h2>동일 프레임 비교: {esc(result['dataset']['preview_frame'])}</h2><div class="previews">{previews}</div>
<h2>측정 조건과 해석</h2><ul>{notes}</ul><p>가중치 SHA-256: {esc(result['environment'].get('checkpoint_sha256') or '해당 없음')}</p></html>'''
    (output / 'report.html').write_text(document, encoding='utf-8')


def benchmark(args):
    cv2.setNumThreads(1)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    emit(status='running', message='기준 프레임을 준비하고 있습니다.', progress=0)
    source, kind, pairs = load_dataset(args)
    torch = model = device = checkpoint = None
    if any(m.startswith('imdn') for m in args.methods):
        emit(status='running', message='IMDN 가중치를 불러오고 있습니다.', progress=2)
        torch, model, device, checkpoint = load_model(args)
    gpu_id = None
    if device is not None and device.type == 'cuda':
        uuid = getattr(torch.cuda.get_device_properties(device), 'uuid', None)
        gpu_id = ('GPU-' + str(uuid).removeprefix('GPU-')) if uuid else None
    sampler = GpuSampler(gpu_id)
    sampler.start()
    raw_rows, results, previews = [], [], []
    preview_index = next((i for i, (_, regions) in enumerate(args.roi_data) if regions), 0)
    preview_source = pairs[preview_index][1]
    image_write(output / 'preview-reference.png', preview_source)
    previews.append({'label': '원본 입력 (고해상도 GT 없음)' if args.mode == 'native' else '기준 영상 (GT)', 'file': 'preview-reference.png'})
    if any('roi' in method for method in args.methods):
        overlay = preview_source.copy()
        for x, y, w, h in args.roi_data[preview_index][0]:
            cv2.rectangle(overlay, (x, y), (x+w-1, y+h-1), (0, 255, 100), 1)
        image_write(output / 'preview-roi-mask.png', overlay)
        previews.append({'label': '엔진 오브젝트 ROI · 녹색 경계 (패딩 전)', 'file': 'preview-roi-mask.png'})
    sync = lambda: torch.cuda.synchronize(device) if device is not None and device.type == 'cuda' else None
    try:
        for method_index, method in enumerate(args.methods):
            emit(status='running', message=f'{LABELS[method]} 워밍업 및 측정 중',
                 progress=5 + method_index / len(args.methods) * 90)
            is_model = method.startswith('imdn')
            is_roi = 'roi' in method
            if is_model:
                infer = model_runner(torch, model, device, half=method.endswith('fp16'))
            else:
                interpolation = {'nearest': cv2.INTER_NEAREST, 'bilinear': cv2.INTER_LINEAR,
                                 'bicubic': cv2.INTER_CUBIC}[method]
                infer = lambda rgb: cv2.resize(rgb, (rgb.shape[1] * args.scale, rgb.shape[0] * args.scale), interpolation=interpolation)
            def evaluate(index, validate=False):
                low = pairs[index][2]
                if is_roi:
                    prediction, flags = selective_upscale(low, args.scale, args.roi_data[index][1],
                                                          lambda crop: infer(crop, validate=validate))
                    execution = 'full_budget' if flags & FRAME_FLAG_BUDGET else 'background' if flags & FRAME_FLAG_BACKGROUND else 'roi'
                    return prediction, execution
                return (infer(low, validate=validate) if is_model else infer(low)), 'full' if is_model else 'interpolation'
            for index in range(len(pairs)):
                for _ in range(args.warmup):
                    evaluate(index, validate=True)
            sync()
            method_rows = []
            start = time.perf_counter()
            repeat = 0
            # A continuous >= 1 s interval makes GPU utilization sampling useful.
            while repeat < args.repeats or time.perf_counter() - start < args.min_seconds:
                for index, (name, _, low) in enumerate(pairs):
                    if is_model:
                        sync()
                    t0 = time.perf_counter_ns()
                    _, execution = evaluate(index)
                    if is_model:
                        sync()
                    elapsed = (time.perf_counter_ns() - t0) / 1_000_000
                    method_rows.append({'method': method, 'frame': name, 'frame_index': index,
                                        'repeat': repeat, 'latency_ms': elapsed, 'execution': execution})
                repeat += 1
                if repeat >= 10000 or time.perf_counter() - start > 300:
                    break
            end = time.perf_counter()
            gpu = sampler.summarize(start, end) if is_model and device.type == 'cuda' else {
                'gpu_utilization_mean_pct': None, 'gpu_memory_peak_mib': None,
                'gpu_samples': 0, 'gpu_unavailable_reason': 'CPU 처리 방식으로 GPU 수치를 비교하지 않습니다.'}
            quality_rows = []
            counts = {}
            for index, (name, reference, low) in enumerate(pairs):
                prediction, execution = evaluate(index, validate=True)
                counts[execution] = counts.get(execution, 0) + 1
                metrics = {} if args.mode == 'native' else {
                    **quality(reference, prediction, args.scale),
                    **roi_quality(reference, prediction, args.roi_data[index][0], args.scale)}
                quality_rows.append({'frame': name, 'execution': execution, **metrics})
                if index == preview_index:
                    filename = f'preview-{method}.png'
                    image_write(output / filename, prediction)
                    previews.append({'label': LABELS[method], 'file': filename})
            execution_summary = (f"ROI {counts.get('roi', 0)} / 전체 전환 {counts.get('full_budget', 0)} / 배경만 {counts.get('background', 0)}"
                                 if is_roi else f"전체 프레임 {len(pairs)}")
            summary = {'method': method, 'label': LABELS[method],
                       'device': str(device) if is_model else 'cpu',
                       **average_quality(quality_rows),
                       'execution_counts': counts, 'execution_summary': execution_summary,
                       **latency_summary([r['latency_ms'] for r in method_rows]), **gpu,
                       'quality_per_frame': quality_rows, 'measurement_seconds': end - start,
                       'measurement_start': start, 'measurement_end': end,
                       'target_duration_met': repeat >= args.repeats and end-start >= args.min_seconds}
            results.append(summary)
            raw_rows.extend(method_rows)
            emit(status='running', message=f'{LABELS[method]} 측정 완료',
                 progress=5 + (method_index + 1) / len(args.methods) * 90, partial=summary)
    finally:
        sampler.stop()
    notes = [
        f'기준 영상에서 OpenCV bicubic 축소로 LR을 만든 ×{args.scale} 재구성 평가입니다. 게임의 원래 고해상도 세부 묘사를 복원했다는 증거는 아닙니다.',
        f'기준 영상 최대 변 길이: {args.max_edge or "원본"} px. 배율의 배수가 되도록 우측·하단을 잘랐습니다. 모든 방식에 같은 LR/기준 영상을 사용했습니다.',
        f'PSNR과 SSIM은 RGB 0–255 기준, 테두리 {args.scale} px 제외 후 프레임별 값의 산술평균입니다. SSIM은 Gaussian σ=1.5, population covariance를 사용합니다.',
        f'프레임마다 {args.warmup}회 워밍업 후 최소 {args.repeats}회 반복, 방식별 최소 {args.min_seconds:g}초 측정합니다. 장면별 크기가 다르면 지연은 혼합 평균입니다.',
        '지연은 메모리의 RGB 입력에서 CPU RGB 출력까지 전처리·추론·후처리를 포함합니다. CUDA 완료를 동기화합니다. 파일 읽기/저장, 지표 계산, 캡처·IPC·화면 표시는 제외합니다.',
        '처리 FPS는 1000 / 평균 지연(ms)입니다. 실제 게임 FPS나 캡처부터 모니터까지의 지연이 아닙니다.',
        'GPU 사용률과 메모리는 nvidia-smi로 약 100 ms 간격으로 수집한 카드 전체 값입니다. 다른 프로세스와 드라이버 표본 구간의 영향을 받으며, 프로세스 전용 수치가 아닙니다. CPU 방식 또는 지원하지 않는 경우 N/A로 표시합니다.',
        '표시한 IMDN은 현재 저장소의 학습 가중치를 실제 실행한 결과입니다. 미구현 모델이나 추론 실패를 보간 결과로 대체하지 않습니다.'
    ]
    if args.mode == 'native':
        notes[:3] = ['원본 프레임을 축소하지 않고 입력합니다. 최대 변 설정은 적용하지 않습니다. 실제 DOOM 입력 크기의 처리 지연을 비교합니다.',
                     '원본 입력보다 큰 정답 고해상도 영상(GT)이 없으므로 전체/ROI PSNR·SSIM은 N/A입니다. 임의 확대 영상을 정답으로 사용하지 않습니다.']
    if any('roi' in method for method in args.methods):
        notes += [
            'ROI는 같은 DOOM 프레임에서 엔진이 제공한 몬스터·아이템 바운딩 박스입니다. 픽셀 단위 분할 마스크가 아니며 배경 일부를 포함할 수 있습니다. 프레임과 ROI를 PNG·해시·좌표로 함께 보존합니다.',
            '라이브 sr_worker.selective_upscale과 동일하게 입력 좌표 12 px 패딩, 겹친 영역 병합, 경계 4×배율 px 혼합을 적용합니다. 배경은 Bicubic입니다. 합친 ROI가 8개 초과 또는 면적 60% 이상이면 전체 SR로 전환합니다.',
            'ROI 화질은 패딩 전 오브젝트 사각형 합집합으로 평가합니다(중복 픽셀 1회). ROI SSIM은 11×11 창이 ROI 내부에 완전히 들어가는 픽셀만 계산합니다. 빈 영역·작은 영역은 해당 지표에서 제외되며, 유효 프레임 수는 JSON에 저장합니다.',
            '축소/복원 모드에서는 좌표를 축소·잘림에 맞춰 보수적으로 변환합니다. 입력 12 px 패딩은 그대로이므로 원본 입력 모드와 전체 SR 전환 비율이 달라질 수 있습니다.',
            f"평가 프레임의 평균 입력 ROI 면적: {np.mean([f['roi_coverage_pct'] for f in args.frame_metadata]):.2f}%, 패딩·병합 후 {np.mean([f['padded_coverage_pct'] for f in args.frame_metadata]):.2f}%. 실제 처리 열은 반복 수가 아닌 고유 장면 수입니다."
        ]
    if any(m.endswith('fp16') for m in args.methods):
        notes.append('IMDN FP16은 합성곱을 FP16으로 실행하고, 채널 평균·분산의 누적 계산은 넘침 방지를 위해 FP32를 사용합니다. 워밍업과 화질 평가에서 NaN/Infinity 출력을 검사합니다.')
    result = {'schema_version': 1, 'run_id': output.name,
              'created_at': datetime.now(timezone.utc).isoformat(),
              'dataset': {'label': kind, 'source': str(source), 'frames': len(pairs),
                          'preview_frame': pairs[preview_index][0], 'images': args.frame_metadata},
              'environment': {'platform': platform.platform(), 'python': platform.python_version(),
                              'opencv': cv2.__version__, 'torch': torch.__version__ if torch else None,
                              'device_name': torch.cuda.get_device_name(device) if device is not None and device.type == 'cuda' else platform.processor() or 'CPU',
                              'checkpoint': str(checkpoint) if checkpoint else None,
                              'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint else None},
              'protocol': {'scale': args.scale, 'mode': args.mode, 'max_edge': args.max_edge, 'warmup': args.warmup,
                           'repeats': args.repeats, 'min_seconds': args.min_seconds,
                           'metrics_color_space': 'RGB', 'crop_border': args.scale, 'notes': notes},
              'results': results, 'previews': previews}
    with (output / 'frames.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(raw_rows[0]))
        writer.writeheader()
        writer.writerows(raw_rows)
    columns = ['label', 'device', 'psnr_db', 'psnr_exact', 'ssim', 'roi_psnr_db', 'roi_psnr_exact', 'roi_ssim', 'roi_metric_frames', 'roi_ssim_frames', 'execution_summary', 'latency_mean_ms',
               'latency_p95_ms', 'processing_fps', 'gpu_utilization_mean_pct',
               'gpu_memory_peak_mib', 'gpu_samples', 'timed_frames']
    with (output / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    with (output / 'gpu-samples.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(['monotonic_seconds', 'gpu_utilization_pct', 'gpu_memory_mib'])
        writer.writerows(sampler.samples)
    write_report(output, result)
    (output / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    emit(status='completed', progress=100, output_dir=str(output), result=result)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', help='기준 이미지 폴더 또는 Avion .json/.bgra 캡처. 기본값: Set5')
    parser.add_argument('--output', required=True, help='새 결과 폴더 (기존 폴더는 덮어쓰지 않음)')
    parser.add_argument('--scale', type=int, choices=[2, 3, 4], default=2)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--checkpoint')
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=['nearest', 'bilinear', 'bicubic', 'imdn'])
    parser.add_argument('--mode', choices=['reconstruction', 'native'], default='reconstruction')
    parser.add_argument('--max-frames', type=int, default=5)
    parser.add_argument('--max-edge', type=int, default=512)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--min-seconds', type=float, default=1)
    args = parser.parse_args(argv)
    if not (1 <= args.max_frames <= 100 and 0 <= args.max_edge <= 4096
            and 1 <= args.warmup <= 30 and 1 <= args.repeats <= 100
            and 0 <= args.min_seconds <= 10) or len(set(args.methods)) != len(args.methods):
        parser.error('프레임 1–100, 최대 크기 0–4096, 워밍업 1–30, 반복 1–100, 최소 시간 0–10 및 중복 없는 방식을 지정하세요.')
    return args


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    try:
        benchmark(parse_args())
    except Exception as exc:
        emit(status='failed', message=f'{type(exc).__name__}: {exc}')
        sys.exit(1)
