"""
XRefine Constants
=================
Shared constants for masking, thresholds, physics values.
모든 매직 넘버를 한 곳에서 관리.
"""

# ==============================================================================
# Masking & Thresholds (Log10 scale)
# ==============================================================================
LOG_MASK_VALUE = -15.0       # Value assigned to masked/invalid regions
LOG_VALID_THRESHOLD = -14.0  # Data above this threshold is considered valid

# ==============================================================================
# Physical Constants
# ==============================================================================
# Cu K-alpha X-ray wavelength [Å]
CU_K_ALPHA_WAVELENGTH = 1.5406

# Classical electron radius [Å]
RE_ANGSTROM = 2.81794032e-5

# ==============================================================================
# Physical Constraints
# ==============================================================================
MIN_THICKNESS = 10.0   # Minimum film thickness [Å]
MAX_ROUGHNESS_RATIO = 0.5  # Roughness must be < 50% of thickness

# ==============================================================================
# Numerical Stability
# ==============================================================================
EPS = 1e-9  # Small value to prevent division by zero
MIN_REFLECTIVITY = 1e-15  # Minimum reflectivity value (clamping)
MAX_REFLECTIVITY = 2.0    # Maximum reflectivity value (clamping)
