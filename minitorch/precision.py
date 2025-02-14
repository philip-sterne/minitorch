import numpy as np

import minitorch.float_precision
import minitorch.ln8_precision

# Global precision class
CURRENT_PRECISION = minitorch.float_precision
CURRENT_TYPE = float


def set_precision(precision: str) -> None:
    """Set the global numeric precision"""
    global CURRENT_PRECISION, CURRENT_TYPE
    if precision.lower() == "float64":
        CURRENT_PRECISION = minitorch.float_precision
        CURRENT_TYPE = float
    elif precision.lower() == "ln8":
        CURRENT_PRECISION = minitorch.ln8_precision
        CURRENT_TYPE = np.int8
    else:
        raise ValueError(f"Unknown precision: {precision}")
