"""Analytical metric checks, capture validation, and a real CLI report round-trip."""
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
import numpy as np
import benchmark as bench


class Metrics(unittest.TestCase):
    def test_half_runner_keeps_original_checkpoint_precision(self):
        import torch
        model = torch.nn.Conv2d(3, 3, 1)
        original = model.weight.detach().clone()
        bench.model_runner(torch, model, torch.device('cpu'), half=True)
        self.assertEqual(model.weight.dtype, torch.float32)
        torch.testing.assert_close(model.weight, original, rtol=0, atol=0)

    def test_half_precision_channel_statistics_do_not_overflow(self):
        import torch
        from IMDN.model.block import mean_channels, stdv_channels
        values = torch.ones((1, 2, 256, 256), dtype=torch.float16) * 4
        values[:, :, :128] = 2
        self.assertTrue(torch.isfinite(mean_channels(values)).all())
        self.assertTrue(torch.isfinite(stdv_channels(values)).all())
        torch.testing.assert_close(mean_channels(values).float(), torch.full((1, 2, 1, 1), 3.0))
        torch.testing.assert_close(stdv_channels(values).float(), torch.ones((1, 2, 1, 1)))

    def test_nonfinite_model_output_is_rejected_before_quantization(self):
        import torch
        class Invalid(torch.nn.Module):
            def forward(self, value):
                return value * float('nan')
        infer = bench.model_runner(torch, Invalid(), torch.device('cpu'))
        with self.assertRaisesRegex(ValueError, 'NaN/Infinity'):
            infer(np.zeros((20, 20, 3), np.uint8), validate=True)

    def test_exact_match_serializes_without_nonstandard_infinity(self):
        image = np.full((32, 32, 3), 100, np.uint8)
        result = bench.quality(image, image.copy(), 2)
        self.assertTrue(result['psnr_exact'])
        self.assertIsNone(result['psnr_db'])
        self.assertAlmostEqual(result['ssim'], 1)
        json.dumps(result, allow_nan=False)

    def test_psnr_known_constant_error(self):
        black = np.zeros((32, 32, 3), np.uint8)
        result = bench.quality(black, black + 10, 0)
        self.assertAlmostEqual(result['psnr_db'], 20 * math.log10(255 / 10), places=10)
        self.assertLess(result['ssim'], 1)

    def test_border_policy_and_size_mismatch(self):
        image = np.zeros((32, 32, 3), np.uint8)
        output = image.copy()
        output[0] = 255
        self.assertTrue(bench.quality(image, output, 2)['psnr_exact'])
        self.assertFalse(bench.quality(image, output, 0)['psnr_exact'])
        with self.assertRaises(ValueError):
            bench.quality(image, output[:20], 2)

    def test_fps_uses_total_time_not_mean_of_instantaneous_rates(self):
        result = bench.latency_summary([10, 30])
        self.assertEqual(result['latency_mean_ms'], 20)
        self.assertEqual(result['processing_fps'], 50)
        self.assertEqual(result['timed_frames'], 2)
        with self.assertRaises(ValueError):
            bench.latency_summary([])

    def test_pair_geometry_and_reference_preservation(self):
        rng = np.random.default_rng(2)
        original = rng.integers(0, 256, (97, 123, 3), dtype=np.uint8)
        reference, low = bench.prepare_pair(original, 4, 0)
        np.testing.assert_array_equal(reference, original[:96, :120])
        self.assertEqual(low.shape, (24, 30, 3))
        reference, _ = bench.prepare_pair(original, 2, 64)
        self.assertEqual(max(reference.shape[:2]), 64)

    def test_missing_gpu_readings_are_not_zero(self):
        sampler = bench.GpuSampler()
        sampler.samples = [(5, 30, 50), (9, 60, 80)]
        self.assertIsNone(sampler.summarize(1, 2)['gpu_utilization_mean_pct'])
        self.assertEqual(sampler.summarize(4, 6)['gpu_utilization_mean_pct'], 30)


class CaptureAndReports(unittest.TestCase):
    def test_padded_capture_and_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            raw = np.zeros((2, 24, 104), np.uint8)
            raw[1, :, :96] = np.tile([10, 20, 30, 255], (24, 24))
            raw.tofile(folder / 'test.bgra')
            meta = {'format': 'avion.lossless.bgra', 'pixelFormat': 'bgra', 'rowOrder': 'top-down',
                    'bytesPerPixel': 4, 'width': 24, 'height': 24, 'frameStrideBytes': 104,
                    'frameSizeBytes': 24 * 104, 'savedFrameCount': 2, 'dataFile': 'test.bgra'}
            source = folder / 'test.json'
            source.write_text(json.dumps(meta))
            frames = list(bench.capture_frames(source, 5))
            self.assertEqual(len(frames), 2)
            np.testing.assert_array_equal(frames[1][1][0, 0], [30, 20, 10])
            meta['savedFrameCount'] = 3
            source.write_text(json.dumps(meta))
            with self.assertRaises(ValueError):
                list(bench.capture_frames(source, 1))

    def test_real_cli_reports_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            source = folder / 'images'
            source.mkdir()
            image = np.random.default_rng(3).integers(0, 256, (48, 64, 3), dtype=np.uint8)
            bench.image_write(source / '테스트.png', image)
            output = folder / 'report'
            command = [sys.executable, '-B', str(Path(bench.__file__)), '--source', str(source),
                       '--output', str(output), '--methods', 'nearest', 'bicubic',
                       '--warmup', '1', '--repeats', '2', '--min-seconds', '0']
            result = subprocess.run(command, capture_output=True, encoding='utf-8', timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((output / 'report.json').read_text(encoding='utf-8'))
            self.assertEqual(len(report['results']), 2)
            self.assertEqual(report['dataset']['frames'], 1)
            self.assertIn('data:image/png;base64', (output / 'report.html').read_text(encoding='utf-8'))
            with (output / 'frames.csv').open(encoding='utf-8-sig') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(float(row['latency_ms']) > 0 for row in rows))
            old = (output / 'report.json').read_bytes()
            repeated = subprocess.run(command, capture_output=True, encoding='utf-8', timeout=60)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertEqual(old, (output / 'report.json').read_bytes())


if __name__ == '__main__':
    unittest.main()
