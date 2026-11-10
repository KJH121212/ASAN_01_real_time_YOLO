"""Progress between fixed start/target values, independent of physical units."""

import math

CALIBRATION_ACTIVATION_FRACTION = .30
CALIBRATION_REARM_FRACTION = .15


def measurement_progress(value, metric_type, start=None, target=None):
    if any(v is None or not math.isfinite(v) for v in (value, start, target)) or start == target:
        return None
    return max(0., min(1., (value-start)/(target-start)))
