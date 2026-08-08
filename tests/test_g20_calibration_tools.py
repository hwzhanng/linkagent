from __future__ import annotations

import unittest

from script.collect_g20_tactile_calibration import _record
from script.fit_g20_tactile_calibration import _fit


class G20CalibrationToolTests(unittest.TestCase):
    def test_capture_normalization_and_affine_fit(self) -> None:
        rows = [
            _record(
                {
                    "finger": "thumb",
                    "raw_sum": raw,
                    "reference_force_n": force,
                    "contact_region": "center",
                },
                index,
            )
            for index, (raw, force) in enumerate(
                ((100.0, 0.0), (200.0, 1.0), (300.0, 2.0), (400.0, 3.0)),
                1,
            )
        ]
        fitted = _fit(rows)
        self.assertAlmostEqual(fitted["gain_n_per_raw_sum"], 0.01)
        self.assertAlmostEqual(fitted["offset_n"], -1.0)
        self.assertAlmostEqual(fitted["rmse_n"], 0.0)
        self.assertAlmostEqual(fitted["r_squared"], 1.0)

    def test_fit_rejects_data_without_zero_and_loaded_samples(self) -> None:
        with self.assertRaisesRegex(ValueError, "near-zero"):
            _fit(
                [
                    {"raw_sum": 100.0, "reference_force_n": 0.5},
                    {"raw_sum": 200.0, "reference_force_n": 1.0},
                    {"raw_sum": 300.0, "reference_force_n": 1.5},
                ]
            )


if __name__ == "__main__":
    unittest.main()
