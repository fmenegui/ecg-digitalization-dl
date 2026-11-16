"""Shared constants: lead vocabulary, physiological constraints and paper conventions."""
from __future__ import annotations

import numpy as np

#: Canonical 12-lead order used for every array in this package.
LEADS: tuple[str, ...] = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
LEAD_INDEX: dict[str, int] = {name: i for i, name in enumerate(LEADS)}
N_LEADS = len(LEADS)

#: Limb leads as linear combinations of the two independent leads (I, II).
#: Einthoven: III = II - I. Goldberger: aVR = -(I+II)/2, aVL = I - II/2, aVF = II - I/2.
LIMB_BASIS = np.array(
    [
        [1.0, 0.0],  # I
        [0.0, 1.0],  # II
        [-1.0, 1.0],  # III
        [-0.5, -0.5],  # aVR
        [1.0, -0.5],  # aVL
        [-0.5, 1.0],  # aVF
    ]
)

#: Major grid spacing of standard ECG paper in millimetres.
MAJOR_GRID_MM = 5.0

#: Standard paper settings. They are defaults, never silently assumed to be measured.
DEFAULT_SPEED_MM_S = 25.0
DEFAULT_GAIN_MM_MV = 10.0

#: Sampling rate of the canonical output contract (matches ``ecgdig``).
OUTPUT_FS = 500.0
