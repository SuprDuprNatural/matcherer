# -*- coding: utf-8 -*-

"""
Matchering - Audio Matching and Mastering Python Library
Copyright (C) 2016-2022 Sergree

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from .log import Code, info, debug, debug_line, ModuleError
from . import Config, Result
from .loader import load
from .stages import main
from .saver import save
from .preview_creator import create_preview
from .utils import get_temp_folder
from .checker import check, check_equality
from .dsp import channel_count, size

import numpy as np
from .analysis import MatcheringAnalysis
from .loader import load
from .checker import check, check_equality
from .stages import __match_levels, __match_frequencies, __correct_levels
from .limiter import limit
from .dsp import size, channel_count
from .results import Result
from .defaults import Config
from .log import Code, info, debug, debug_line, ModuleError

def analyze(
    target_file: str,
    reference_file: str,
    config: Config = Config(),
    allow_limiter: bool = True
) -> MatcheringAnalysis:
    """
    Analyze the TARGET and REFERENCE, returning a 'MatcheringAnalysis'
    that holds all EQ and amplitude coefficients needed to apply
    the same processing to other audio.
    """
    debug_line()
    info(Code.INFO_LOADING)

    temp_folder = config.temp_folder

    target, target_sr = load(target_file, "target", temp_folder)
    target, target_sr = check(target, target_sr, config, "target")

    reference, reference_sr = load(reference_file, "reference", temp_folder)
    reference, reference_sr = check(reference, reference_sr, config, "reference")

    if not config.allow_equality:
        check_equality(target, reference)

    if (
        not (target_sr == reference_sr == config.internal_sample_rate)
        or not (channel_count(target) == channel_count(reference) == 2)
        or not (size(target) > config.fft_size and size(reference) > config.fft_size)
    ):
        raise ModuleError(Code.ERROR_VALIDATION)

    # Match levels
    (
        target_mid,
        target_side,
        final_amp_coef,
        tgt_mid_loudest,
        tgt_side_loudest,
        ref_mid_loudest,
        ref_side_loudest,
        divisions,
        piece_size,
        reference_match_rms
    ) = __match_levels(target, reference, config)

    # Match frequencies
    (
        result_no_limiter,
        result_no_limiter_mid,
        mid_fir,
        side_fir
    ) = __match_frequencies(
        target_mid, target_side,
        tgt_mid_loudest, ref_mid_loudest,
        tgt_side_loudest, ref_side_loudest,
        config
    )

    # Correct final levels
    result_no_limiter = __correct_levels(
        result_no_limiter,
        result_no_limiter_mid,
        divisions,
        piece_size,
        reference_match_rms,
        config
    )

    # Return a full MatcheringAnalysis
    return MatcheringAnalysis(
        mid_fir=mid_fir,
        side_fir=side_fir,
        final_amplitude_coefficient=final_amp_coef,
        reference_match_rms=reference_match_rms,
        need_limiter=allow_limiter
    )

def apply_analysis(
    input_file: str,
    analysis: MatcheringAnalysis,
    config: Config = Config(),
    output_file: str = None
):
    """
    Apply the same 'analysis' to a brand-new input audio.
    This mimics the final stages of the pipeline: convolving
    with the stored FIR, matching amplitude, possibly limiting,
    then saving.
    """

    debug_line()
    info(Code.INFO_MATCHING_LEVELS)  # or a more appropriate code
    temp_folder = config.temp_folder

    # 1) Load the new audio
    new_audio, sr = load(input_file, "target", temp_folder)  # "target" for convenience
    new_audio, sr = check(new_audio, sr, config, "target")

    # 2) Convert L/R --> M/S so we can apply the stored FIR
    from .dsp import lr_to_ms, ms_to_lr, amplify
    mid, side = lr_to_ms(new_audio)

    # 3) Convolve with the stored mid_fir, side_fir
    from scipy.signal import fftconvolve
    mid_convolved = fftconvolve(mid, analysis.mid_fir, mode="same")
    side_convolved = fftconvolve(side, analysis.side_fir, mode="same")

    # 4) Convert back to L/R
    result_no_limiter = ms_to_lr(mid_convolved, side_convolved)

    # 5) Apply amplitude coefficient
    if not np.isclose(analysis.final_amplitude_coefficient, 1.0):
        result_no_limiter = amplify(result_no_limiter, analysis.final_amplitude_coefficient)

    # 6) Optionally apply the limiter if analysis says so
    if analysis.need_limiter:
        from .limiter import limit
        result = limit(result_no_limiter, config)
    else:
        result = result_no_limiter

    # 7) Save or return the processed audio
    if output_file:
        from .saver import save
        # For example default to 16-bit PCM
        save(
            file=output_file,
            result=result,
            sample_rate=config.internal_sample_rate,
            subtype="PCM_16",
            name="final result"
        )
        return None
    else:
        # Return the raw float numpy array for further usage
        return result

def process(
    target: str,
    reference: str,
    results: list,
    config: Config = Config(),
    preview_target: Result = None,
    preview_result: Result = None,
):
    debug(
        "Please give us a star to help the project: https://github.com/sergree/matchering"
    )
    debug_line()
    info(Code.INFO_LOADING)

    if not results:
        raise RuntimeError(f"The result list is empty")

    # Get a temporary folder for converting mp3's
    temp_folder = config.temp_folder if config.temp_folder else get_temp_folder(results)

    # Load the target
    target, target_sample_rate = load(target, "target", temp_folder)
    # Analyze the target
    target, target_sample_rate = check(target, target_sample_rate, config, "target")

    # Load the reference
    reference, reference_sample_rate = load(reference, "reference", temp_folder)
    # Analyze the reference
    reference, reference_sample_rate = check(
        reference, reference_sample_rate, config, "reference"
    )

    # Analyze the target and the reference together
    if not config.allow_equality:
        check_equality(target, reference)

    # Validation of the most important conditions
    if (
        not (target_sample_rate == reference_sample_rate == config.internal_sample_rate)
        or not (channel_count(target) == channel_count(reference) == 2)
        or not (size(target) > config.fft_size and size(reference) > config.fft_size)
    ):
        raise ModuleError(Code.ERROR_VALIDATION)

    # Process
    result, result_no_limiter, result_no_limiter_normalized, mid_fir, side_fir = main(
        target,
        reference,
        config,
        need_default=any(rr.use_limiter for rr in results),
        need_no_limiter=any(not rr.use_limiter and not rr.normalize for rr in results),
        need_no_limiter_normalized=any(
            not rr.use_limiter and rr.normalize for rr in results
        ),
    )

    del reference
    if not (preview_target or preview_result):
        del target

    debug_line()
    info(Code.INFO_EXPORTING)

    # Save
    for required_result in results:
        if required_result.use_limiter:
            correct_result = result
        else:
            if required_result.normalize:
                correct_result = result_no_limiter_normalized
            else:
                correct_result = result_no_limiter
        save(
            required_result.file,
            correct_result,
            config.internal_sample_rate,
            required_result.subtype,
        )

    # Creating a preview (if needed)
    if preview_target or preview_result:
        result = next(
            item
            for item in [result, result_no_limiter, result_no_limiter_normalized]
            if item is not None
        )
        create_preview(target, result, config, preview_target, preview_result)

    debug_line()
    info(Code.INFO_COMPLETED)
