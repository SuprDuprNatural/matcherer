# matchering/analysis.py

import numpy as np

class MatcheringAnalysis:
    """
    A container that holds all parameters needed to apply
    the same EQ / amplitude / limiter steps to any new file.
    """

    def __init__(
        self,
        mid_fir: np.ndarray,
        side_fir: np.ndarray,
        final_amplitude_coefficient: float,
        reference_match_rms: float,
        need_limiter: bool
    ):
        self.mid_fir = mid_fir
        self.side_fir = side_fir
        self.final_amplitude_coefficient = final_amplitude_coefficient
        self.reference_match_rms = reference_match_rms
        self.need_limiter = need_limiter  # You can store more flags if you like!

    def __repr__(self):
        return (
            f"MatcheringAnalysis(\n"
            f"  mid_fir_len={len(self.mid_fir)},\n"
            f"  side_fir_len={len(self.side_fir)},\n"
            f"  final_amp_coef={self.final_amplitude_coefficient},\n"
            f"  ref_match_rms={self.reference_match_rms},\n"
            f"  need_limiter={self.need_limiter}\n"
            f")"
        )