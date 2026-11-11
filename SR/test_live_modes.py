"""Exercise the actual live worker protocol and Electron's method environment."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np
import torch
import sr_worker as worker

ROOT = Path(__file__).resolve().parent


class LiveModes(unittest.TestCase):
    def run_worker(self, method, *, scale=2, roi=False, device='cpu', checkpoint=None):
        image = np.random.default_rng(10).integers(0, 256, (96, 96, 3), dtype=np.uint8)
        pixels, width, height, pitch = worker.rgb_to_argb8888_bytes(image)
        regions = [(30, 30, 12, 12)] if roi else []
        request = worker.HEADER.pack(worker.ROI_REQUEST_MAGIC if roi else worker.REQUEST_MAGIC,
                                     width, height, pitch, 1, len(regions), len(pixels))
        request += b''.join(worker.RECT.pack(*r) for r in regions) + pixels
        with tempfile.TemporaryDirectory(prefix='avion-live-mode-') as temporary:
            status = Path(temporary) / 'status.json'
            env = {key: value for key, value in os.environ.items()
                   if not key.upper().startswith(('AVION_SR', 'AVION_ROI_RECORD_'))}
            env.update(AVION_SR_METHOD=method, AVION_SR_DEVICE=device, AVION_SR_STATUS_FILE=str(status))
            args = [sys.executable, '-B', str(ROOT / 'sr_worker.py'), '--scale', str(scale)]
            if checkpoint:
                args += ['--checkpoint', checkpoint]
            result = subprocess.run(args, input=request, capture_output=True, env=env, timeout=90)
            state = json.loads(status.read_text()) if status.exists() else None
        return result, state, image

    def decode(self, result, scale):
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors='replace'))
        magic, width, height, pitch, fmt, flags, size = worker.HEADER.unpack_from(result.stdout)
        self.assertEqual((magic, width, height, fmt), (worker.RESPONSE_MAGIC, 96 * scale, 96 * scale, 1))
        self.assertEqual(len(result.stdout), worker.HEADER.size + size)
        return worker.argb8888_bytes_to_rgb(result.stdout[worker.HEADER.size:], width, height, pitch), flags

    def test_interpolation_exact_output_without_loading_model(self):
        for method, algorithm, scale in [('nearest', cv2.INTER_NEAREST, 2),
                                          ('bilinear', cv2.INTER_LINEAR, 3),
                                          ('bicubic', cv2.INTER_CUBIC, 4)]:
            with self.subTest(method=method):
                result, state, image = self.run_worker(method, scale=scale, checkpoint='missing-model.pth')
                output, flags = self.decode(result, scale)
                np.testing.assert_array_equal(output, cv2.resize(image, (96 * scale, 96 * scale), interpolation=algorithm))
                self.assertEqual(flags, worker.FRAME_FLAG_FALLBACK)
                self.assertEqual(state, {'status': 'ready', 'method': method, 'scale': scale, 'device': 'cpu'})
                self.assertNotIn(b'loaded IMDN', result.stderr)

    def test_explicit_fp32_full_and_roi(self):
        for method, roi, flags in [('imdn', False, 1), ('imdn-roi', True, 5)]:
            with self.subTest(method=method):
                result, state, _ = self.run_worker(method, roi=roi)
                _, actual = self.decode(result, 2)
                self.assertEqual(actual, flags)
                self.assertEqual(state['status'], 'ready')
                self.assertIn(b'precision: fp32', result.stderr)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for live FP16')
    def test_actual_cuda_fp16_full_and_roi(self):
        for method, roi, flags in [('imdn-fp16', False, 1), ('imdn-roi-fp16', True, 5)]:
            with self.subTest(method=method):
                result, state, _ = self.run_worker(method, roi=roi, device='cuda')
                output, actual = self.decode(result, 2)
                self.assertEqual(actual, flags)
                self.assertEqual(state['device'], 'cuda')
                self.assertIn(b'precision: fp16', result.stderr)
                self.assertGreater(output.std(), 1)

    def test_selected_model_failure_is_reported_instead_of_bicubic(self):
        result, state, _ = self.run_worker('imdn', checkpoint='missing-model.pth')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b'')
        self.assertEqual(state['status'], 'failed')

    def test_fp16_cpu_and_roi_without_metadata_are_rejected(self):
        for method, message in [('imdn-fp16', 'CUDA'), ('imdn-roi', 'metadata')]:
            with self.subTest(method=method):
                result, state, _ = self.run_worker(method)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, b'')
                self.assertEqual(state['status'], 'failed')
                self.assertIn(message, state['message'])


if __name__ == '__main__':
    unittest.main()
