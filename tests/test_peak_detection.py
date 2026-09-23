import unittest

import numpy as np

from models.peak_detection import (
    calculate_heart_rate,
    calculate_interval_cv,
    classify_s1_s2,
    detect_peaks,
    get_s1_intervals,
)


class HeartTimingTests(unittest.TestCase):
    def test_detector_keeps_s1_and_s2_less_than_point_three_seconds_apart(self):
        envelope = np.zeros(2000)
        envelope[[100, 320, 900, 1120, 1700, 1920]] = 1
        envelope[600] = 0.1
        peaks, _ = detect_peaks(envelope, sample_rate=1000)
        np.testing.assert_array_equal(peaks, [100, 320, 900, 1120, 1700, 1920])
        quiet_peaks, _ = detect_peaks(envelope * 0.001, sample_rate=1000)
        np.testing.assert_array_equal(quiet_peaks, peaks)
        self.assertEqual(len(detect_peaks(np.zeros(1000), sample_rate=1000)[0]), 0)

    def test_classifies_sequences_starting_on_either_heart_sound(self):
        s1_first = np.array([0, 200, 800, 1000, 1600, 1800, 2400, 2600, 3200])
        for peaks, first_label in ((s1_first, "S1"), (s1_first[1:], "S2")):
            times, labels = classify_s1_s2(peaks, sample_rate=1000)
            self.assertEqual(labels[0], first_label)
            self.assertEqual(labels[1], "S2" if first_label == "S1" else "S1")
            self.assertEqual(calculate_heart_rate(times, labels), 75.0)

    def test_ambiguous_or_short_sequences_have_no_timing_estimate(self):
        sequences = (
            np.array([0, 400, 800, 1200, 1600, 2000, 2400]),
            np.array([0, 200, 800, 1000]),
        )
        for peaks in sequences:
            times, labels = classify_s1_s2(peaks, sample_rate=1000)
            self.assertEqual(labels, ["Unclassified"] * len(peaks))
            self.assertIsNone(calculate_heart_rate(times, labels))
            self.assertIsNone(calculate_interval_cv(times, labels))

    def test_missed_sound_keeps_only_a_consistent_cycle_run(self):
        peaks = np.array([0, 200, 800, 1000, 1600, 2400, 2600, 3200, 3400, 4000])
        times, labels = classify_s1_s2(peaks, sample_rate=1000)
        self.assertIn("Unclassified", labels)
        np.testing.assert_allclose(get_s1_intervals(times, labels), [.8, .8])
        self.assertEqual(calculate_heart_rate(times, labels), 75.0)
        self.assertIsNone(calculate_interval_cv(times, labels))

    def test_interval_statistics_need_enough_complete_beats(self):
        times = np.array([0, .2, .8, 1.0, 1.8, 2.0, 2.8, 3.0])
        labels = ["S1", "S2"] * 4
        np.testing.assert_allclose(get_s1_intervals(times, labels), [.8, 1.0, 1.0])
        self.assertEqual(calculate_heart_rate(times, labels), 60.0)
        self.assertAlmostEqual(
            calculate_interval_cv(times, labels),
            np.std([.8, 1.0, 1.0], ddof=1) / np.mean([.8, 1.0, 1.0]),
        )
        self.assertIsNone(calculate_interval_cv(times[:6], labels[:6]))

    def test_invalid_s1_timing_is_rejected(self):
        self.assertEqual(len(get_s1_intervals([0, .2, .1], ["S1", "S2", "S1"])), 0)
        self.assertEqual(len(get_s1_intervals([0, np.nan, .8], ["S1", "S2", "S1"])), 0)
        self.assertEqual(len(get_s1_intervals([0, .2, .8], ["S1", "Unclassified", "S1"])), 0)
        self.assertEqual(len(get_s1_intervals([0, .2], ["S1"])), 0)
        with self.assertRaises(ValueError):
            classify_s1_s2([0, 200], sample_rate=0)


if __name__ == "__main__":
    unittest.main()
