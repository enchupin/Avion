"""Lossless DOOM inputs paired with engine-provided sprite rectangles."""
import hashlib
import json
import math
import os
from pathlib import Path
import time

import cv2
import numpy as np

FORMAT = 'avion.doom.roi.v1'
MANIFEST = 'roi-manifest.json'


def validate_regions(regions, width, height):
    if not isinstance(regions, list) or len(regions) > 64:
        raise ValueError('ROI 목록이 없거나 64개 제한을 초과했습니다.')
    for rect in regions:
        if (not isinstance(rect, (list, tuple)) or len(rect) != 4
                or any(type(v) is not int for v in rect)):
            raise ValueError('ROI 좌표는 정수 x, y, width, height여야 합니다.')
        x, y, w, h = rect
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
            raise ValueError('ROI가 프레임 범위를 벗어났습니다.')
    return regions


def remap_regions(regions, factor_x, factor_y, width, height):
    """Outward rounding, followed by clipping to the cropped destination."""
    mapped = []
    for x, y, w, h in regions:
        left, top = max(0, math.floor(x * factor_x)), max(0, math.floor(y * factor_y))
        right, bottom = min(width, math.ceil((x + w) * factor_x)), min(height, math.ceil((y + h) * factor_y))
        if right > left and bottom > top:
            mapped.append([left, top, right - left, bottom - top])
    return mapped


def region_mask(regions, width, height):
    mask = np.zeros((height, width), np.uint8)
    for x, y, w, h in regions:
        mask[y:y+h, x:x+w] = 1
    return mask.astype(bool)


class RoiRecorder:
    def __init__(self, directory, interval_ms=500, max_frames=60):
        self.directory = Path(directory)
        self.interval = max(0, int(interval_ms)) / 1000
        self.limit = min(100, max(1, int(max_frames)))
        self.last = None
        self.frames = []

    @classmethod
    def from_environment(cls):
        directory = os.environ.get('AVION_ROI_RECORD_DIR')
        return cls(directory, os.environ.get('AVION_ROI_RECORD_INTERVAL_MS', 500),
                   os.environ.get('AVION_ROI_RECORD_MAX_FRAMES', 60)) if directory else None

    def record(self, rgb, regions):
        now = time.perf_counter()
        if len(self.frames) >= self.limit or (self.last is not None and now - self.last < self.interval):
            return False
        height, width = rgb.shape[:2]
        validate_regions(regions, width, height)
        if not self.frames:
            self.directory.mkdir(parents=True, exist_ok=False)
        name = f'frame-{len(self.frames)+1:06}.png'
        ok, encoded = cv2.imencode('.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise ValueError('DOOM 프레임 PNG 인코딩 실패')
        encoded.tofile(self.directory / name)
        self.frames.append({'file': name, 'width': width, 'height': height,
                            'regions': [list(r) for r in regions], 'monotonic_seconds': now,
                            'rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest()})
        data = {'format': FORMAT, 'source': 'DOOM engine sprite bounds: monsters and pickups',
                'coordinates': 'source pixels, xywh, right/bottom exclusive', 'frames': self.frames}
        temporary = self.directory / 'roi-manifest.tmp'
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(self.directory / MANIFEST)
        self.last = now
        return True


def read_frames(source, count):
    source = Path(source)
    manifest = source / MANIFEST if source.is_dir() else source
    if manifest.stat().st_size > 2 * 1024 ** 2:
        raise ValueError('ROI 메타데이터가 너무 큽니다.')
    data = json.loads(manifest.read_text(encoding='utf-8-sig'))
    if data.get('format') != FORMAT or not isinstance(data.get('frames'), list) or not 1 <= len(data['frames']) <= 100:
        raise ValueError('유효한 DOOM ROI 데이터가 아닙니다.')
    seen = set()
    for frame in data['frames']:
        file = frame.get('file', '')
        if (not file or file in seen or Path(file).name != file or '/' in file or '\\' in file
                or (manifest.parent / file).resolve().parent != manifest.parent.resolve()):
            raise ValueError('ROI 프레임 파일 경로가 올바르지 않습니다.')
        seen.add(file)
        w, h = frame.get('width'), frame.get('height')
        if type(w) is not int or type(h) is not int or not (1 <= w <= 4096 and 1 <= h <= 4096):
            raise ValueError('ROI 프레임 크기가 올바르지 않습니다.')
        validate_regions(frame.get('regions'), w, h)
    selected = np.linspace(0, len(data['frames'])-1, min(count, len(data['frames'])), dtype=int)
    for index in selected:
        frame = data['frames'][index]
        rgb = cv2.imdecode(np.fromfile(manifest.parent / frame['file'], np.uint8), cv2.IMREAD_COLOR)
        if rgb is None or rgb.shape[:2] != (frame['height'], frame['width']):
            raise ValueError('ROI 프레임과 메타데이터의 크기가 일치하지 않습니다.')
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        if hashlib.sha256(rgb.tobytes()).hexdigest() != frame.get('rgb_sha256'):
            raise ValueError('ROI 프레임 해시가 일치하지 않습니다.')
        yield frame['file'], rgb, frame['regions']
