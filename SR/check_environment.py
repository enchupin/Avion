"""JSON-only environment probe used by the Electron setup panel."""
import importlib
import json
from pathlib import Path
import platform
import struct
import sys

sys.dont_write_bytecode = True
result = {
    'python': sys.executable, 'version': platform.python_version(),
    'compatible': (3, 10) <= sys.version_info[:2] < (3, 14) and struct.calcsize('P') == 8,
    'packages': {}, 'errors': [], 'cuda': False, 'device': 'CPU', 'model': False,
    'isVenv': sys.prefix != sys.base_prefix,
    'gpuMessage': 'PyTorch 설치 후 GPU 실행 가능 여부를 확인할 수 있습니다.',
}
for name, minimum in [('numpy', (1, 24)), ('cv2', (4, 8)), ('skimage', (0, 21)), ('torch', (2, 1))]:
    try:
        module = importlib.import_module(name)
        version = module.__version__
        if tuple(int(part) for part in version.split('.')[:2]) < minimum:
            raise RuntimeError(f'{version}: version too old (requires {minimum})')
        result['packages'][name] = version
    except Exception as exc:
        result['packages'][name] = None
        result['errors'].append(f'{name}: {exc}')

if result['packages'].get('torch'):
    try:
        import torch
        result['cudaBuild'] = torch.version.cuda
        if torch.cuda.is_available():
            # Device enumeration alone does not prove that this GPU's kernels work.
            with torch.inference_mode():
                layer = torch.nn.Conv2d(3, 8, 3).cuda()
                layer(torch.zeros(1, 3, 16, 16, device='cuda')).cpu()
            result['cuda'] = True
            result['device'] = torch.cuda.get_device_name(0)
            result['gpuMessage'] = f"사용 가능 · {result['device']} · CUDA {torch.version.cuda}"
        else:
            result['gpuMessage'] = ('사용 불가 · CPU용 PyTorch가 설치되어 있습니다.' if not torch.version.cuda
                                    else '사용 불가 · CUDA GPU 또는 호환 NVIDIA 드라이버를 찾을 수 없습니다.')
    except Exception as exc:
        result['cuda'] = False
        result['gpuMessage'] = f'사용 불가 · GPU 실행 오류: {exc}'

if result['compatible'] and not result['errors']:
    try:
        import numpy as np
        import torch
        from sr_worker import load_imdn, selective_upscale, FRAME_FLAG_MODEL, FRAME_FLAG_ROI
        torch.set_num_threads(2)
        for scale in (2, 3, 4):
            checkpoint = Path(__file__).parent / 'IMDN' / 'checkpoints' / f'IMDN_x{scale}.pth'
            _, model, _ = load_imdn(str(checkpoint), scale, 'cpu', True)
            with torch.inference_mode():
                output = model(torch.zeros(1, 3, 16, 16))
            assert output.shape == (1, 3, 16 * scale, 16 * scale)
            assert torch.isfinite(output).all()
        # Exercise the same ROI composition path used by the live worker.
        from sr_worker import model_upscale
        image, flags = selective_upscale(np.zeros((64, 64, 3), dtype=np.uint8), 4,
            [(24, 24, 8, 8)], lambda crop: model_upscale(torch, model, torch.device('cpu'), crop))
        assert image.shape == (256, 256, 3) and flags == FRAME_FLAG_MODEL | FRAME_FLAG_ROI
        result['model'] = True
        if result['cuda']:
            # Test IMDN as well, using the actual worker's GPU inference path.
            try:
                gpu_output = model_upscale(torch, model.to('cuda'), torch.device('cuda'), np.zeros((16, 16, 3), dtype=np.uint8))
                assert gpu_output.shape == (64, 64, 3)
            except Exception as exc:
                result['cuda'] = False
                result['gpuMessage'] = f'사용 불가 · GPU IMDN 추론 실패: {exc}'
    except Exception as exc:
        result['errors'].append(f'IMDN/ROI: {exc}')

if result['cuda'] and not result['model']:
    result['gpuMessage'] = f"GPU 인식됨 · {result['device']} · SR 패키지 설치 후 IMDN 검증이 필요합니다."
    result['cuda'] = False
result['ready'] = result['compatible'] and result['model'] and not result['errors']
print(json.dumps(result, ensure_ascii=True))
