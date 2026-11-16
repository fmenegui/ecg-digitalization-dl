"""ecgdl: hybrid deep-learning + classical-geometry digitizer for paper ECG images."""
from .constants import LEADS
from .pipeline import DigitizationResult, Digitizer, DigitizerConfig

__version__ = "0.1.0"
__all__ = ["LEADS", "DigitizationResult", "Digitizer", "DigitizerConfig", "__version__"]
