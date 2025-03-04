import minitorch.float_precision
import minitorch.ln8_precision

# Global precision class
# CURRENT_PRECISION = minitorch.float_precision
CURRENT_PRECISION = minitorch.ln8_precision


# def set_precision(precision: str) -> None:
#     """Set the global numeric precision"""
#     global CURRENT_PRECISION
#     if precision.lower() == "float64":
#         CURRENT_PRECISION = minitorch.float_precision
#     elif precision.lower() == "ln8":
#         CURRENT_PRECISION = minitorch.ln8_precision
#     else:
#         raise ValueError(f"Unknown precision: {precision}")
