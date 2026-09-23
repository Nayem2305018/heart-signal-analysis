import unittest

import numpy as np

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
