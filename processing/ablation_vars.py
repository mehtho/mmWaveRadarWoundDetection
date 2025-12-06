from dataclasses import dataclass


@dataclass
class AblationVars:
    MIN_BIN: int = 7
    MAX_BIN: int = 13

    # Existing normalization flag
    NORM_PER_FRAME: bool = False

    # Wubtract per-subject dry baseline (all pos)
    SUBTRACT_BASELINE: bool = True 

    # found in testing this is bad, dont use
    ONLY_CENTER_POS: bool = False

    NORM_PER_POSITION = True


# Global config object used by other modules:
ABLATION_VARS = AblationVars()
