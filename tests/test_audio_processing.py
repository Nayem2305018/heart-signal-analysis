import unittest
import io

import numpy as np
import soundfile as sf

import audio_processing as dsp


class AudioProcessingTests(unittest.TestCase):
    def setUp(self):
        self.rate = 16000
        time = np.arange(self.rate) / self.rate
        self.tone = (0.4 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)

    def test_read_audio_and_export(self):
        encoded = dsp.wav_bytes(self.tone, self.rate)
        decoded, rate = dsp.read_audio(encoded)
        self.assertEqual(rate, self.rate)
        self.assertEqual(len(decoded), len(self.tone))
        self.assertLess(np.max(np.abs(decoded - self.tone)), 0.001)
        with self.assertRaises(ValueError):
            dsp.read_audio(dsp.wav_bytes(self.tone[:100], self.rate))

    def test_browser_playback_upsamples_low_rate_without_changing_duration(self):
        original_rate = 2000  # The bundled heart sound recordings use this rate.
        low_rate_tone = self.tone[::self.rate // original_rate]
        playback = dsp.playback_wav_bytes(low_rate_tone, original_rate)
        info = sf.info(io.BytesIO(playback))
        self.assertEqual(info.samplerate, dsp.MIN_PLAYBACK_RATE)
        self.assertEqual(info.subtype, "PCM_16")
        self.assertAlmostEqual(info.duration, len(low_rate_tone) / original_rate,
                               places=3)
        self.assertEqual(dsp.playback_wav_bytes(self.tone, self.rate),
                         dsp.wav_bytes(self.tone, self.rate))

    def test_five_minute_limit(self):
        rate = 1000
        five_minutes = np.zeros(5 * 60 * rate, dtype=np.float32)
        decoded, decoded_rate = dsp.read_audio(dsp.wav_bytes(five_minutes, rate))
        self.assertEqual(decoded_rate, rate)
        self.assertEqual(len(decoded), len(five_minutes))
        with self.assertRaisesRegex(ValueError, "5 minutes"):
            dsp.read_audio(dsp.wav_bytes(np.zeros(301 * rate, dtype=np.float32), rate))

    def test_long_recording_preview_is_bounded(self):
        rate = 1000
        time = np.arange(121 * rate) / rate
        audio = (0.2 * np.sin(2 * np.pi * 120 * time)).astype(np.float32)
        frequencies, times, magnitude = dsp.spectrogram_preview(audio, rate)
        self.assertEqual(magnitude.shape, (len(frequencies), len(times)))
        self.assertLessEqual(len(times), 900)
        self.assertGreater(times[-1], 119)
        _, _, transformed = dsp.stft(audio, rate)
        restored = dsp.istft(transformed, rate, len(audio))
        self.assertEqual(len(restored), len(audio))
        self.assertLess(np.max(np.abs(restored - audio)), 1e-4)

    def test_spectral_edit_preserves_length_and_reduces_selected_tone(self):
        edited = dsp.paint_spectrogram(self.tone, self.rate, 0, 1, 350, 550, 0.9)
        self.assertEqual(len(edited), len(self.tone))
        self.assertLess(np.sqrt(np.mean(edited ** 2)),
                        0.5 * np.sqrt(np.mean(self.tone ** 2)))

    def test_frequency_filter_keeps_selected_bands(self):
        time = np.arange(self.rate) / self.rate
        tones = {frequency: np.sin(2 * np.pi * frequency * time)
                 for frequency in (200, 2000, 6000)}
        mixed = sum(0.2 * tone for tone in tones.values()).astype(np.float32)

        def level(samples, frequency):
            return abs(2 * np.dot(samples, tones[frequency]) / len(samples))

        cases = [
            ("Low pass", 500, None, 200, 6000),
            ("High pass", 4000, None, 6000, 200),
            ("Band pass", 1000, 3000, 2000, 200),
            ("Band stop", 1000, 3000, 200, 2000),
        ]
        for kind, low, high, kept, removed in cases:
            with self.subTest(kind=kind):
                filtered = dsp.frequency_filter(mixed, self.rate, kind, low, high)
                self.assertEqual(len(filtered), len(mixed))
                self.assertGreater(level(filtered, kept), 0.14)
                self.assertLess(level(filtered, removed), 0.06)
        with self.assertRaises(ValueError):
            dsp.frequency_filter(mixed, self.rate, "Band pass", 3000, 1000)

    def test_comparison_alignment_overlap_and_metrics(self):
        rate = 1000
        rng = np.random.default_rng(7)
        first = np.repeat(rng.uniform(0.1, 0.8, 500), 10).astype(np.float32)
        second = np.r_[np.zeros(350, dtype=np.float32), first]
        estimate = dsp.estimate_alignment(first, second, rate, max_shift_seconds=1)
        self.assertIsNotNone(estimate)
        self.assertAlmostEqual(estimate[0], -0.35, delta=0.01)
        self.assertGreater(estimate[1], 0.99)
        reverse = dsp.estimate_alignment(second, first, rate, max_shift_seconds=1)
        self.assertAlmostEqual(reverse[0], 0.35, delta=0.01)

        aligned_first, aligned_second, start = dsp.comparison_overlap(first, second, -350)
        self.assertEqual(start, 0)
        self.assertTrue(np.array_equal(aligned_first, aligned_second))
        rms_difference, correlation, level_difference = dsp.comparison_stats(
            aligned_first, aligned_second)
        self.assertAlmostEqual(rms_difference, 0)
        self.assertAlmostEqual(correlation, 1)
        self.assertAlmostEqual(level_difference, 0)
        self.assertIsNone(dsp.estimate_alignment(np.zeros_like(first), second, rate))

        shifted_first, shifted_second, start = dsp.comparison_overlap(first, first[300:], 300)
        self.assertEqual(start, 300)
        self.assertTrue(np.array_equal(shifted_first, shifted_second))
        empty_first, empty_second, _ = dsp.comparison_overlap(first, second, len(first))
        self.assertEqual(len(empty_first), 0)
        self.assertEqual(len(empty_second), 0)
        with self.assertRaises(ValueError):
            dsp.comparison_stats(empty_first, empty_second)
        _, correlation, level_difference = dsp.comparison_stats(first, first * 0.5)
        self.assertAlmostEqual(correlation, 1)
        self.assertAlmostEqual(level_difference, -6.0206, places=3)

    def test_pitch_and_delay(self):
        pitch = dsp.detect_pitch(self.tone, self.rate, 0.5)
        self.assertIsNotNone(pitch)
        self.assertEqual(pitch[1], "A4")
        delayed = dsp.effects(self.tone, self.rate, "Delay", 0.5, 0.25)
        self.assertEqual(len(delayed), len(self.tone) + self.rate // 4)
        self.assertTrue(np.allclose(delayed[:self.rate // 4], self.tone[:self.rate // 4]))

    def test_spectrum_uses_the_whole_recording(self):
        time = np.arange(self.rate) / self.rate
        later_tone = (0.4 * np.sin(2 * np.pi * 880 * time)).astype(np.float32)
        frequencies, power = dsp.spectrum_db(np.concatenate((self.tone, later_tone)),
                                              self.rate)
        level_440 = power[np.argmin(np.abs(frequencies - 440))]
        level_880 = power[np.argmin(np.abs(frequencies - 880))]
        self.assertGreater(level_440, -35)
        self.assertGreater(level_880, -35)

    def test_processing_outputs_are_finite(self):
        for output in (dsp.denoise(self.tone, self.rate, 1.5),
                       dsp.equalize(self.tone, self.rate, 3, -2, 4),
                       dsp.room_simulate(self.tone, self.rate, "Hall", 0.5)[0]):
            self.assertTrue(np.isfinite(output).all())
            self.assertGreaterEqual(len(output), len(self.tone))


if __name__ == "__main__":
    unittest.main()
