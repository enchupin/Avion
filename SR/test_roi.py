"""Deterministic ROI compositing and real worker protocol checks."""
import importlib.util
from pathlib import Path
import struct
import subprocess
import sys
import unittest

sys.dont_write_bytecode = True

import numpy as np

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('sr_worker', ROOT / 'sr_worker.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class ROIProcessing(unittest.TestCase):
    def test_merge_transitive_and_clip(self):
        boxes = worker.prepare_regions([(0, 0, 8, 8), (20, 0, 8, 8),
                                        (10, 0, 8, 8)], 32, 20, margin=2)
        self.assertEqual(boxes, [(0, 0, 30, 10)])

    def test_background_skips_inference(self):
        image = np.zeros((200, 320, 3), dtype=np.uint8)
        def forbidden(_):
            self.fail('background frame must not run the model')
        output, flags = worker.selective_upscale(image, 2, [], forbidden)
        self.assertEqual(output.shape, (400, 640, 3))
        self.assertEqual(flags, worker.FRAME_FLAG_BACKGROUND)

    def test_crop_interior_background_and_feather(self):
        image = np.zeros((200, 320, 3), dtype=np.uint8)
        shapes = []
        def infer(crop):
            shapes.append(crop.shape)
            return np.full((crop.shape[0]*2, crop.shape[1]*2, 3), 200, np.uint8)
        output, flags = worker.selective_upscale(image, 2, [(100, 70, 20, 30)], infer)
        self.assertEqual(shapes, [(54, 44, 3)])
        self.assertTrue(np.all(output[140:200, 200:240] == 200))
        self.assertTrue(np.all(output[:100] == 0))
        self.assertGreater(output[116, 176, 0], 0)
        self.assertLess(output[116, 176, 0], 200)
        self.assertEqual(flags, worker.FRAME_FLAG_MODEL | worker.FRAME_FLAG_ROI)

    def test_budget_switches_to_full_frame(self):
        image = np.zeros((200, 320, 3), dtype=np.uint8)
        shapes = []
        def infer(crop):
            shapes.append(crop.shape)
            return worker.fallback_upscale(crop, 2, 'nearest')
        _, flags = worker.selective_upscale(image, 2, [(0, 0, 320, 200)], infer)
        self.assertEqual(shapes, [image.shape])
        self.assertTrue(flags & worker.FRAME_FLAG_BUDGET)

    def test_missing_model_is_explicit(self):
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        _, flags = worker.selective_upscale(image, 2, [(5, 5, 5, 5)], None)
        self.assertEqual(flags, worker.FRAME_FLAG_FALLBACK)

    def test_real_worker_mixed_protocol(self):
        # Model-required integration: checks actual crop inference, empty ROI,
        # budget fallback and legacy full-frame requests in one persistent worker.
        header = worker.HEADER
        w, h = 96, 96
        pixels = bytes(w*h*4)
        requests = [(worker.ROI_REQUEST_MAGIC, [(30, 30, 12, 12)]),
                    (worker.ROI_REQUEST_MAGIC, []),
                    (worker.ROI_REQUEST_MAGIC, [(0, 0, w, h)]),
                    (worker.REQUEST_MAGIC, [])]
        data = b''.join(header.pack(magic, w, h, w*4, 1, len(rois), len(pixels))
                        + b''.join(worker.RECT.pack(*r) for r in rois) + pixels
                        for magic, rois in requests)
        result = subprocess.run([sys.executable, str(ROOT/'sr_worker.py'),
                                 '--strict-model'], input=data, capture_output=True,
                                timeout=90)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors='replace'))
        cursor = 0
        flags = []
        for _ in requests:
            magic, ow, oh, pitch, fmt, flag, size = header.unpack_from(result.stdout, cursor)
            self.assertEqual((magic, ow, oh, pitch, fmt, size),
                             (worker.RESPONSE_MAGIC, w*2, h*2, w*8, 1, w*h*16))
            cursor += header.size + size
            flags.append(flag)
        self.assertEqual(cursor, len(result.stdout))
        self.assertEqual(flags, [5, 8, 17, 1])


if __name__ == '__main__':
    unittest.main()
