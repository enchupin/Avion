"""Headless integration check: two renderers, complete SR roundtrip, fallback.

Run from the repository root: python Doom/quickcheck/check_sr_windows.py
Uses a short generated demo and a controlled worker with a 40 ms delay.
"""
import os
import math
from pathlib import Path
import re
import subprocess
import struct
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
EXE = ROOT / 'Doom/build-local/src/Release/chocolate-doom.exe'
IWAD = ROOT / 'Doom/iwads/freedoom-0.13.0/freedoom-0.13.0/freedoom2.wad'
WORKER = '''import struct, sys, time
h = struct.Struct('<7I')
def read(n):
    b = bytearray()
    while len(b) < n:
        chunk = sys.stdin.buffer.read(n-len(b))
        if not chunk: return None
        b.extend(chunk)
    return b
while True:
    raw = read(h.size)
    if raw is None: break
    magic,w,height,pitch,fmt,_,size = h.unpack(raw)
    if magic == 0x32525341:
        rects = read(_ * 16)
        assert rects is not None
        for x,y,rw,rh in struct.iter_unpack('<4I', rects):
            assert rw > 0 and rh > 0 and x+rw <= w and y+rh <= height
    data = read(size)
    if data is None: break
    time.sleep(0.04)
    sys.stdout.buffer.write(h.pack(0x31525352,w,height,pitch,fmt,2,size))
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()
'''

with tempfile.TemporaryDirectory(prefix='avion-sr-') as folder:
    temp = Path(folder)
    demo = temp / 'short.lmp'
    demo.write_bytes(bytes([109, 2, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0])
                     + bytes([0, 0, 0, 0]) * 12 + bytes([128]))
    worker = temp / 'worker.py'
    worker.write_text(WORKER)
    # Deterministic map fixture: retain MAP01 geometry, replace THINGS with
    # the player and a pickup 128 units in front. No monster AI affects bounds.
    wad = IWAD.read_bytes()
    _, count, directory = struct.unpack_from('<4sII', wad)
    lumps = [struct.unpack_from('<II8s', wad, directory+i*16) for i in range(count)]
    start = next(i for i, (_, _, name) in enumerate(lumps) if name.rstrip(b'\0') == b'MAP01')
    map_lumps = [(name, wad[offset:offset+size]) for offset,size,name in lumps[start:start+11]]
    things = list(struct.iter_unpack('<hhhhh', map_lumps[1][1]))
    player = next(t for t in things if t[3] == 1)
    x,y,angle,_,options = player
    pickup = (x+round(128*math.cos(math.radians(angle))),
              y+round(128*math.sin(math.radians(angle))), 0, 2011, 7)
    map_lumps[1] = (map_lumps[1][0], struct.pack('<hhhhh', *player) + struct.pack('<hhhhh', *pickup))
    payload = bytearray()
    entries = bytearray()
    for name, content in map_lumps:
        entries.extend(struct.pack('<II8s', 12+len(payload), len(content), name))
        payload.extend(content)
    fixture = temp / 'roi-fixture.wad'
    fixture.write_bytes(struct.pack('<4sII', b'PWAD', len(map_lumps), 12+len(payload))+payload+entries)
    env = {key.upper(): value for key, value in os.environ.items()}
    env.update(SDL_VIDEODRIVER='dummy', SDL_RENDER_DRIVER='software',
               SDL_AUDIODRIVER='dummy', AVION_SR_PYTHON=sys.executable,
               AVION_SR_WORKER=str(worker), AVION_SR='1')
    base = [str(EXE), '-iwad', str(IWAD), '-window', '-nosound', '-nojoy',
            '-nograbmouse', '-srnoconsole', '-srlatency', '-timedemo', str(demo),
            '-config', str(temp / 'default.cfg'),
            '-extraconfig', str(temp / 'extra.cfg')]
    for name, extra in [('dual', ['-srcompare']), ('single', []),
                         ('roi', ['-srcompare', '-srroi', '-srroi-debug', '-file', str(fixture)]),
                         ('disabled', ['-srcompare', '-nosr'])]:
        ticks = 60 if name == 'roi' else 12
        demo.write_bytes(bytes([109, 2, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0])
                         + bytes(4*ticks) + bytes([128]))
        result = subprocess.run(base + extra, env=env, cwd=ROOT,
                                capture_output=True, timeout=45)
        output = (result.stdout + result.stderr).decode(errors='replace')
        # Doom timedemo deliberately exits via I_Error after printing results.
        assert f'{ticks} gametics' in output, output
        samples = re.findall(r'SR latency \[FALLBACK\] total=([\d.]+) ms.*?roundtrip\+SR=([\d.]+)', output)
        if name == 'disabled':
            assert not samples, output
        else:
            assert samples, output
            for total, roundtrip in samples:
                assert float(total) >= float(roundtrip) >= 35, output
        if name == 'roi':
            counts = [int(n) for n in re.findall(r'raw regions=(\d+)', output)]
            assert counts and max(counts) > 0, output
        print(f'{name}: PASS, latency samples={samples}')
    if '--model' in sys.argv:
        env['AVION_SR_WORKER'] = str(ROOT / 'SR/sr_worker.py')
        env['AVION_SR_CHECKPOINT'] = str(ROOT / 'SR/IMDN/checkpoints/IMDN_x2.pth')
        env['AVION_SR_SCALE'] = '2'
        result = subprocess.run(base + ['-srcompare'], env=env, cwd=ROOT,
                                capture_output=True, timeout=90)
        output = (result.stdout + result.stderr).decode(errors='replace')
        assert '12 gametics' in output and 'SR latency [MODEL]' in output, output
        assert 'SR latency [FALLBACK]' not in output, output
        print('real model: PASS')
        for line in output.splitlines():
            if 'SR latency [MODEL]' in line or 'loaded IMDN' in line:
                print(line)
        result = subprocess.run(base + ['-srcompare', '-srroi', '-file', str(fixture)],
                                env=env, cwd=ROOT, capture_output=True, timeout=90)
        output = (result.stdout + result.stderr).decode(errors='replace')
        assert '12 gametics' in output and 'SR experiment: ROI model' in output, output
        assert 'SR latency [FALLBACK]' not in output, output
        print('real model + engine ROI: PASS')
