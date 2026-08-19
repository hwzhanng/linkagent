"""Raw G20 tactile-matrix features shared by recording and future policies.

The G20 SDK exposes five 12x6 uint8 matrices.  This module preserves those
raw units: it does not claim that a taxel value is a force in newtons.  A
fresh zero matrix is kept distinct from an invalid or stale finger.

The released Physics-v1 Actor still consumes its frozen five calibrated
pressure values.  These 50 features are an independent data/research path.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import median
from typing import Sequence

from proto import TactileFrame


FINGER_COUNT = 5
TAXEL_ROWS = 12
TAXEL_COLUMNS = 6
FEATURES_PER_FINGER = 10
TACTILE_MATRIX_FEATURE_DIM = FINGER_COUNT * FEATURES_PER_FINGER
TACTILE_MATRIX_FEATURE_NAMES = (
    "total_load_norm",
    "peak_taxel_norm",
    "contact_area_fraction",
    "centroid_row_norm",
    "centroid_column_norm",
    "spread_row_norm",
    "spread_column_norm",
    "load_rate_norm",
    "valid",
    "stale_or_invalid",
)

RawFingerMatrix = tuple[tuple[float, ...], ...]
RawTactileMatrices = tuple[RawFingerMatrix, ...]
TaxelMask = tuple[tuple[bool, ...], ...]
TactileMask = tuple[TaxelMask, ...]


def _finite(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _matrix(value: Sequence[Sequence[float]], *, field_name: str) -> RawFingerMatrix:
    rows = tuple(tuple(row) for row in value)
    if len(rows) != TAXEL_ROWS or any(len(row) != TAXEL_COLUMNS for row in rows):
        raise ValueError(f"{field_name} must have shape {TAXEL_ROWS}x{TAXEL_COLUMNS}")
    return tuple(
        tuple(_finite(item, field_name=f"{field_name}[{row_index}][{column_index}]")
              for column_index, item in enumerate(row))
        for row_index, row in enumerate(rows)
    )


def _matrices(value: Sequence[Sequence[Sequence[float]]], *, field_name: str) -> RawTactileMatrices:
    fingers = tuple(value)
    if len(fingers) != FINGER_COUNT:
        raise ValueError(f"{field_name} must contain {FINGER_COUNT} finger matrices")
    return tuple(
        _matrix(item, field_name=f"{field_name}[{index}]")
        for index, item in enumerate(fingers)
    )


def _mask(value: Sequence[Sequence[Sequence[bool]]], *, field_name: str) -> TactileMask:
    fingers = tuple(value)
    if len(fingers) != FINGER_COUNT:
        raise ValueError(f"{field_name} must contain {FINGER_COUNT} finger masks")
    result: list[TaxelMask] = []
    for finger_index, finger in enumerate(fingers):
        rows = tuple(tuple(row) for row in finger)
        if len(rows) != TAXEL_ROWS or any(len(row) != TAXEL_COLUMNS for row in rows):
            raise ValueError(f"{field_name}[{finger_index}] must have shape 12x6")
        if any(not isinstance(item, bool) for row in rows for item in row):
            raise TypeError(f"{field_name} values must be bools")
        result.append(rows)
    return tuple(result)


def _zero_matrices() -> RawTactileMatrices:
    return tuple(
        tuple(tuple(0.0 for _ in range(TAXEL_COLUMNS)) for _ in range(TAXEL_ROWS))
        for _ in range(FINGER_COUNT)
    )


def _false_mask() -> TactileMask:
    return tuple(
        tuple(tuple(False for _ in range(TAXEL_COLUMNS)) for _ in range(TAXEL_ROWS))
        for _ in range(FINGER_COUNT)
    )


@dataclass(frozen=True, slots=True)
class TactileMatrixBaseline:
    """Per-taxel no-contact baseline and unusable-taxel mask, in raw units."""

    baseline_raw: RawTactileMatrices
    noise_mad_raw: RawTactileMatrices
    bad_taxel_mask: TactileMask

    def __post_init__(self) -> None:
        object.__setattr__(self, "baseline_raw", _matrices(self.baseline_raw, field_name="baseline_raw"))
        object.__setattr__(self, "noise_mad_raw", _matrices(self.noise_mad_raw, field_name="noise_mad_raw"))
        object.__setattr__(self, "bad_taxel_mask", _mask(self.bad_taxel_mask, field_name="bad_taxel_mask"))

    @classmethod
    def zero(cls) -> "TactileMatrixBaseline":
        return cls(_zero_matrices(), _zero_matrices(), _false_mask())


@dataclass(frozen=True, slots=True)
class TactileMatrixFeatureConfig:
    maximum_age_s: float = 0.15
    contact_taxel_threshold_raw: float = 1.0
    total_load_full_scale_raw: float = 1000.0
    peak_taxel_full_scale_raw: float = 100.0
    load_rate_full_scale_raw_per_s: float = 5000.0

    def __post_init__(self) -> None:
        for field_name in (
            "maximum_age_s",
            "contact_taxel_threshold_raw",
            "total_load_full_scale_raw",
            "peak_taxel_full_scale_raw",
            "load_rate_full_scale_raw_per_s",
        ):
            value = _finite(getattr(self, field_name), field_name=field_name)
            if value <= 0.0:
                raise ValueError(f"{field_name} must be positive")
            object.__setattr__(self, field_name, value)


@dataclass(frozen=True, slots=True)
class FingerMatrixFeatures:
    total_load_raw: float
    peak_taxel_raw: float
    contact_area_fraction: float
    centroid_row_norm: float
    centroid_column_norm: float
    spread_row_norm: float
    spread_column_norm: float
    load_rate_norm: float
    valid: bool

    def as_vector(self, config: TactileMatrixFeatureConfig) -> tuple[float, ...]:
        if not self.valid:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        return (
            min(1.0, self.total_load_raw / config.total_load_full_scale_raw),
            min(1.0, self.peak_taxel_raw / config.peak_taxel_full_scale_raw),
            self.contact_area_fraction,
            self.centroid_row_norm,
            self.centroid_column_norm,
            self.spread_row_norm,
            self.spread_column_norm,
            self.load_rate_norm,
            1.0,
            0.0,
        )


@dataclass(frozen=True, slots=True)
class TactileMatrixFeatureFrame:
    source_sequence: int
    feature_extractor_id: str
    fingers: tuple[FingerMatrixFeatures, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.fingers) != FINGER_COUNT:
            raise ValueError("fingers must contain five feature records")
        if len(self.values) != TACTILE_MATRIX_FEATURE_DIM:
            raise ValueError(f"values must contain {TACTILE_MATRIX_FEATURE_DIM} features")

    @property
    def valid_mask(self) -> tuple[bool, ...]:
        return tuple(item.valid for item in self.fingers)

    @property
    def contact_mask(self) -> tuple[bool, ...]:
        return tuple(item.valid and item.contact_area_fraction > 0.0 for item in self.fingers)


def fit_tactile_matrix_baseline(
    no_contact_samples: Sequence[Sequence[Sequence[Sequence[float]]]],
    *,
    maximum_noise_mad_raw: float | None = None,
) -> TactileMatrixBaseline:
    """Fit a robust median/MAD baseline from at least five no-contact frames."""

    samples = tuple(_matrices(item, field_name=f"no_contact_samples[{index}]")
                    for index, item in enumerate(no_contact_samples))
    if len(samples) < 5:
        raise ValueError("at least five no-contact samples are required")
    if maximum_noise_mad_raw is not None:
        maximum_noise_mad_raw = _finite(maximum_noise_mad_raw, field_name="maximum_noise_mad_raw")
        if maximum_noise_mad_raw <= 0.0:
            raise ValueError("maximum_noise_mad_raw must be positive")

    baselines: list[list[list[float]]] = []
    deviations: list[list[list[float]]] = []
    masks: list[list[list[bool]]] = []
    for finger in range(FINGER_COUNT):
        baseline_rows: list[list[float]] = []
        deviation_rows: list[list[float]] = []
        mask_rows: list[list[bool]] = []
        for row in range(TAXEL_ROWS):
            baseline_row: list[float] = []
            deviation_row: list[float] = []
            mask_row: list[bool] = []
            for column in range(TAXEL_COLUMNS):
                values = [sample[finger][row][column] for sample in samples]
                center = float(median(values))
                mad = float(median(abs(value - center) for value in values))
                baseline_row.append(center)
                deviation_row.append(mad)
                mask_row.append(maximum_noise_mad_raw is not None and mad > maximum_noise_mad_raw)
            baseline_rows.append(baseline_row)
            deviation_rows.append(deviation_row)
            mask_rows.append(mask_row)
        baselines.append(baseline_rows)
        deviations.append(deviation_rows)
        masks.append(mask_rows)
    return TactileMatrixBaseline(
        _matrices(baselines, field_name="baseline_raw"),
        _matrices(deviations, field_name="noise_mad_raw"),
        _mask(masks, field_name="bad_taxel_mask"),
    )


class TactileMatrixFeatureExtractor:
    """Stateful 5x12x6 -> 50D extractor with an explicit raw-unit contract."""

    feature_extractor_id = "g20_raw_matrix_50d_v1"

    def __init__(
        self,
        config: TactileMatrixFeatureConfig | None = None,
        baseline: TactileMatrixBaseline | None = None,
    ) -> None:
        self.config = config or TactileMatrixFeatureConfig()
        self.baseline = baseline or TactileMatrixBaseline.zero()
        self.reset()

    def reset(self) -> None:
        self._previous_total_load_raw = (0.0,) * FINGER_COUNT
        self._previous_received_ns: int | None = None

    def extract(self, frame: TactileFrame, *, now_ns: int) -> TactileMatrixFeatureFrame:
        if isinstance(now_ns, bool) or not isinstance(now_ns, int) or now_ns < 0:
            raise ValueError("now_ns must be an integer >= 0")
        if self._previous_received_ns is None or frame.received_monotonic_ns <= self._previous_received_ns:
            dt_s: float | None = None
        else:
            dt_s = (frame.received_monotonic_ns - self._previous_received_ns) / 1.0e9

        results: list[FingerMatrixFeatures] = []
        for finger_index, finger in enumerate(frame.fingers):
            sample_ns = finger.matrix_sample_monotonic_ns
            age_s = (now_ns - sample_ns) / 1.0e9 if sample_ns is not None else math.inf
            valid = (
                finger.matrix_valid
                and finger.matrix_weight_raw is not None
                and 0.0 <= age_s <= self.config.maximum_age_s
            )
            if not valid:
                results.append(FingerMatrixFeatures(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False))
                continue

            corrected: list[tuple[float, float, float]] = []
            active_count = 0
            good_count = 0
            for row in range(TAXEL_ROWS):
                row_norm = -1.0 + 2.0 * row / (TAXEL_ROWS - 1)
                for column in range(TAXEL_COLUMNS):
                    if self.baseline.bad_taxel_mask[finger_index][row][column]:
                        continue
                    column_norm = -1.0 + 2.0 * column / (TAXEL_COLUMNS - 1)
                    raw = float(finger.matrix_weight_raw[row][column])
                    value = max(0.0, raw - self.baseline.baseline_raw[finger_index][row][column])
                    corrected.append((value, row_norm, column_norm))
                    good_count += 1
                    active_count += value >= self.config.contact_taxel_threshold_raw

            total = sum(item[0] for item in corrected)
            peak = max((item[0] for item in corrected), default=0.0)
            area = active_count / max(1, good_count)
            if total > 0.0:
                centroid_row = sum(value * row for value, row, _ in corrected) / total
                centroid_column = sum(value * column for value, _, column in corrected) / total
                spread_row = math.sqrt(sum(value * (row - centroid_row) ** 2 for value, row, _ in corrected) / total)
                spread_column = math.sqrt(sum(value * (column - centroid_column) ** 2 for value, _, column in corrected) / total)
            else:
                centroid_row = centroid_column = spread_row = spread_column = 0.0
            load_rate = 0.0 if dt_s is None else (
                (total - self._previous_total_load_raw[finger_index])
                / dt_s
                / self.config.load_rate_full_scale_raw_per_s
            )
            results.append(
                FingerMatrixFeatures(
                    total,
                    peak,
                    min(1.0, area),
                    max(-1.0, min(1.0, centroid_row)),
                    max(-1.0, min(1.0, centroid_column)),
                    max(0.0, min(1.0, spread_row)),
                    max(0.0, min(1.0, spread_column)),
                    max(-1.0, min(1.0, load_rate)),
                    True,
                )
            )

        result_tuple = tuple(results)
        values = tuple(value for item in result_tuple for value in item.as_vector(self.config))
        self._previous_total_load_raw = tuple(item.total_load_raw for item in result_tuple)
        self._previous_received_ns = frame.received_monotonic_ns
        return TactileMatrixFeatureFrame(
            source_sequence=frame.source_sequence,
            feature_extractor_id=self.feature_extractor_id,
            fingers=result_tuple,
            values=values,
        )


__all__ = [
    "FEATURES_PER_FINGER",
    "FINGER_COUNT",
    "TACTILE_MATRIX_FEATURE_DIM",
    "TACTILE_MATRIX_FEATURE_NAMES",
    "TAXEL_COLUMNS",
    "TAXEL_ROWS",
    "FingerMatrixFeatures",
    "TactileMatrixBaseline",
    "TactileMatrixFeatureConfig",
    "TactileMatrixFeatureExtractor",
    "TactileMatrixFeatureFrame",
    "fit_tactile_matrix_baseline",
]
