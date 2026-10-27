"""Analytical ROI metrics, capture pairing and live pipeline equivalence."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import benchmark as bench
from roi_dataset import RoiRecorder, read_frames, remap_regions, region_mask
from sr_worker import selective_upscale, FRAME_FLAG_BUDGET, FRAME_FLAG_BACKGROUND


class RoiEvaluation(unittest.TestCase):
    def test_recording_roundtrip_integrity_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / 'scenes'
            image = np.random.default_rng(8).integers(0, 256, (40, 60, 3), np.uint8)
            recorder = RoiRecorder(folder, interval_ms=0, max_frames=2)
            self.assertTrue(recorder.record(image, [(4, 6, 18, 20)]))
            self.assertTrue(recorder.record(image, []))
            self.assertFalse(recorder.record(image, []))
            frames = list(read_frames(folder, 5))
            np.testing.assert_array_equal(frames[0][1], image)
            self.assertEqual(frames[0][2], [[4, 6, 18, 20]])
            self.assertEqual(frames[1][2], [])
            with self.assertRaises(FileExistsError):
                RoiRecorder(folder).record(image, [])
            bench.image_write(folder / frames[0][0], image + np.uint8(1))
            with self.assertRaisesRegex(ValueError, '해시'):
                list(read_frames(folder, 5))

    def test_invalid_missing_regions_and_paths_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / 'scenes'
            RoiRecorder(folder).record(np.zeros((40, 60, 3), np.uint8), [])
            manifest = folder / 'roi-manifest.json'
            original = json.loads(manifest.read_text())
            for key, bad in [('file', '../outside.png'), ('regions', None), ('regions', [[59, 0, 2, 2]]), ('regions', [[0.5, 1, 2, 2]])]:
                data = json.loads(json.dumps(original))
                data['frames'][0][key] = bad
                manifest.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    list(read_frames(folder, 5))

    def test_roi_union_psnr_and_ssim_exclude_background(self):
        reference = np.full((64, 64, 3), 100, np.uint8)
        output = reference.copy()
        output[:20] = 255
        regions = [[20, 25, 25, 25], [25, 30, 20, 20]]
        result = bench.roi_quality(reference, output, regions, 2)
        self.assertEqual(result['roi_pixels'], 625)
        self.assertTrue(result['roi_psnr_exact'])
        self.assertAlmostEqual(result['roi_ssim'], 1)
        output[25:50, 20:45] += 10
        result = bench.roi_quality(reference, output, regions, 2)
        self.assertAlmostEqual(result['roi_psnr_db'], 20 * np.log10(255 / 10))
        self.assertEqual(result['roi_ssim_pixels'], 225)
        tiny = bench.roi_quality(reference, output, [[20, 25, 2, 2]], 2)
        self.assertIsNone(tiny['roi_ssim'])
        empty = bench.roi_quality(reference, output, [], 2)
        self.assertIsNone(empty['roi_psnr_db'])
        self.assertEqual(bench.average_quality([result, empty])['roi_metric_frames'], 1)

    def test_coordinate_resizing_clips_modcrop_without_stretching(self):
        regions = [[116, 92, 7, 5], [2, 2, 21, 21]]
        mapped = remap_regions(regions, 1/4, 1/4, 30, 24)
        self.assertEqual(mapped, [[29, 23, 1, 1], [0, 0, 6, 6]])
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / 'scenes'
            image = np.zeros((97, 123, 3), np.uint8)
            RoiRecorder(folder).record(image, regions)
            args = bench.parse_args(['--source', str(folder), '--output', str(Path(temp)/'report'), '--scale', '4', '--max-edge', '0', '--methods', 'imdn-roi'])
            _, _, pairs = bench.load_dataset(args)
            self.assertEqual(pairs[0][2].shape, (24, 30, 3))
            self.assertEqual(args.roi_data[0][1], mapped)
            args.mode = 'native'
            _, _, pairs = bench.load_dataset(args)
            np.testing.assert_array_equal(pairs[0][2], image)
            self.assertEqual(args.roi_data[0][1], regions)
            self.assertIsNone(args.frame_metadata[0]['reference_sha256'])

    def test_live_roi_model_budget_and_background_paths(self):
        image = np.full((100, 160, 3), 20, np.uint8)
        calls = []
        def infer(crop):
            calls.append(crop.shape)
            return np.full((crop.shape[0]*2, crop.shape[1]*2, 3), 100, np.uint8)
        out, flags = selective_upscale(image, 2, [], infer)
        self.assertEqual(flags, FRAME_FLAG_BACKGROUND)
        self.assertEqual(calls, [])
        out, flags = selective_upscale(image, 2, [[40, 40, 20, 20]], infer)
        self.assertFalse(flags & FRAME_FLAG_BUDGET)
        self.assertEqual(calls[-1], (44, 44, 3))
        self.assertTrue(np.all(out[80:120, 80:120] == 100))
        self.assertTrue(np.all(out[:10, :10] == 20))
        out, flags = selective_upscale(image, 2, [[0, 0, 160, 100]], infer)
        self.assertTrue(flags & FRAME_FLAG_BUDGET)
        self.assertEqual(calls[-1], image.shape)

    def test_roi_method_requires_real_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            args = bench.parse_args(['--output', str(Path(temp)/'out'), '--methods', 'imdn-roi'])
            with self.assertRaisesRegex(ValueError, 'roi-manifest'):
                bench.load_dataset(args)


if __name__ == '__main__':
    unittest.main()
