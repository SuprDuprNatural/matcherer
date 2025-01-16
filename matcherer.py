# matcherer.py

import argparse
import os
import sys
import json
import numpy as np
import soundfile as sf

from pathlib import Path
from datetime import datetime
from scipy.signal import fftconvolve, freqz
import matplotlib.pyplot as plt

import matchering as mg
from matchering.core import process as old_process  # if needed
from matchering.log import Code, info, debug_line
from matchering.dsp import lr_to_ms, ms_to_lr, amplify
from matchering.saver import save as saver_save

# --------------------------------------------------------------------
# 1) Customized Logging
# --------------------------------------------------------------------

def log_info(text):
    print(f"{datetime.now()}: INFO: {text}")

def log_warning(text):
    print(f"{datetime.now()}: WARNING: {text}")

mg.log(info_handler=log_info, warning_handler=log_warning)

# --------------------------------------------------------------------
# 2) Analysis I/O (JSON)
# --------------------------------------------------------------------

def save_analysis(analysis_obj, path):
    """
    Save the analysis object (FIR, amplitude, etc.) as JSON.
    """
    data = {
        "final_amplitude_coefficient": float(analysis_obj.final_amplitude_coefficient),
        "reference_match_rms": float(analysis_obj.reference_match_rms) if analysis_obj.reference_match_rms else None,
        "need_limiter": bool(analysis_obj.need_limiter),
                "mid_fir": analysis_obj.mid_fir.tolist(),
        "side_fir": analysis_obj.side_fir.tolist(),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[OK] Analysis saved to '{path}'")

def load_analysis(path):
    """
    Load the analysis object (FIR, amplitude, etc.) from JSON.
    """
    with open(path, "r") as f:
        data = json.load(f)

    # Reconstruct a MatcheringAnalysis instance
    analysis_obj = mg.analysis.MatcheringAnalysis(
        mid_fir=np.array(data["mid_fir"], dtype=np.float32),
        side_fir=np.array(data["side_fir"], dtype=np.float32),
        final_amplitude_coefficient=data["final_amplitude_coefficient"],
        reference_match_rms=data["reference_match_rms"],
        need_limiter=data["need_limiter"]
    )
    print(f"[OK] Analysis loaded from '{path}'")
    return analysis_obj

# --------------------------------------------------------------------
# 3) Helpers
# --------------------------------------------------------------------

def find_audio_files(directory):
    """
    Find all audio files in a given directory.
    """
    audio_extensions = [".wav", ".mp3", ".flac", ".ogg", ".aac", ".m4a"]
    return [
        str(f) for f in Path(directory).glob("**/*")
        if f.suffix.lower() in audio_extensions
    ]

def confirm_combinations(combinations):
    """
    Display combinations (basenames only) and get user confirmation.
    """
    print("\nThe following combinations will be processed:")
    for idx, (target, reference) in enumerate(combinations, start=1):
        print(f"{idx}: {Path(target).stem} -> {Path(reference).stem}")
    response = input("\nDo you want to proceed with these combinations? (y/n): ").strip().lower()
    return response == "y"

def save_fir_ir(fir_coeffs, sample_rate, output_path, normalize=False):
    """
    Writes a 1D array 'fir_coeffs' as a mono WAV IR in 32-bit float format.
    """
    # Optionally normalize (commented out for now)
    # peak = np.max(np.abs(fir_coeffs))
    # if normalize and peak > 1.0:
    #     fir_coeffs = fir_coeffs / peak

    mono_data = fir_coeffs.reshape(-1, 1)
    sf.write(output_path, mono_data, samplerate=sample_rate, subtype="PCM_32")
    print(f"[OK] Wrote mono IR to '{output_path}' (sample_rate={sample_rate} Hz)")

def save_fir_data(fir_coeffs, sample_rate, output_dir, file_prefix="fir", save_plot=True):
    """
    Save FIR filter data as a TXT file and optionally plot the frequency response as a PNG.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save coefficients to a TXT file
    txt_file = os.path.join(output_dir, f"{file_prefix}_coefficients.txt")
    np.savetxt(txt_file, fir_coeffs, fmt="%.8f")
    print(f"[OK] Saved FIR coefficients to {txt_file}")

    if save_plot:
        w, h = freqz(fir_coeffs, worN=4096)
        freqs = w * sample_rate / (2 * np.pi)
        magnitude_db = 20 * np.log10(np.abs(h))

        plt.figure(figsize=(8, 5))
        plt.plot(freqs, magnitude_db, label="Filter Response (dB)")
        plt.title(f"{file_prefix} FIR Frequency Response")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Gain (dB)")
        plt.ylim([-30, 30])
        plt.xscale("log")
        plt.grid(True)
        plt.legend()

        png_file = os.path.join(output_dir, f"{file_prefix}_frequency_response.png")
        plt.savefig(png_file)
        plt.close()
        print(f"[OK] Saved FIR frequency response plot to {png_file}")

# --------------------------------------------------------------------
# 4) Matchering Analysis and Application
# --------------------------------------------------------------------

def analyze(
    target_file: str,
    reference_file: str,
    config: mg.Config = mg.Config(),
    allow_limiter: bool = True
) -> "mg.analysis.MatcheringAnalysis":
    """
    Analyze the TARGET and REFERENCE, returning a 'MatcheringAnalysis'
    that holds all EQ (FIR) and amplitude parameters for re-use.
    """
    debug_line()
    info(Code.INFO_LOADING)

    analysis_obj = mg.analyze(
        target_file=target_file,
        reference_file=reference_file,
        config=config,
        allow_limiter=allow_limiter
    )
    return analysis_obj

def apply_analysis(
    input_file: str,
    analysis: "mg.analysis.MatcheringAnalysis",
    config: mg.Config = mg.Config(),
    output_file: str = None,
    bit_depth: str = "FLOAT"
):
    """
    Apply an existing 'analysis' to a brand-new input audio.
    """
    debug_line()
    info(Code.INFO_MATCHING_LEVELS)

    temp_folder = config.temp_folder
    new_audio, sr = mg.load(input_file, "target", temp_folder)
    new_audio, sr = mg.check(new_audio, sr, config, "target")

    # Convert to M/S
    mid, side = lr_to_ms(new_audio)

    # Convolve
    mid_convolved = fftconvolve(mid, analysis.mid_fir, mode="same")
    side_convolved = fftconvolve(side, analysis.side_fir, mode="same")

    # Convert back to L/R
    result_no_limiter = ms_to_lr(mid_convolved, side_convolved)

    # Apply amplitude
    if not np.isclose(analysis.final_amplitude_coefficient, 1.0):
        result_no_limiter = amplify(result_no_limiter, analysis.final_amplitude_coefficient)

    # Optional limiter
    result = result_no_limiter
    if analysis.need_limiter:
        result = mg.limiter.limit(result_no_limiter, config)

    # Choose bit depth
    if bit_depth == 16:
        subtype = "PCM_16"
    elif bit_depth == 24:
        subtype = "PCM_24"
    elif bit_depth == "float":
        subtype = "FLOAT"
    else:
        subtype = "PCM_16"  # fallback

    if output_file:
        saver_save(output_file, result, config.internal_sample_rate, subtype)
    else:
        return result

def matcher_audio_combinations(
    combinations,
    output_dir,
    bit_depths,
    plot_fir=False,
    analysis_file=None,
    reuse_analysis=False
):
    """
    Process all combinations of target and reference audio files,
    optionally reusing or saving analysis to a JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)

    for target, reference in combinations:
        target_basename = Path(target).stem
        reference_basename = Path(reference).stem

        # If reusing an existing analysis file, load once per combination.
        if reuse_analysis and analysis_file and os.path.isfile(analysis_file):
            log_info(f"Loading analysis from '{analysis_file}'...")
            analysis_obj = load_analysis(analysis_file)
        else:
            log_info(f"Analyzing target='{target}' reference='{reference}' ...")
            analysis_obj = analyze(target, reference)
            # Always save new analysis if we have a path
            if analysis_file:
                save_analysis(analysis_obj, analysis_file)
            else:
                # or auto-generate a name
                default_json = os.path.join(
                    output_dir,
                    f"{target_basename}_{reference_basename}_analysis.json"
                )
                save_analysis(analysis_obj, default_json)

        # Show summary
        log_info(f"Analysis Results:\n  final_amp_coef={analysis_obj.final_amplitude_coefficient}\n"
                 f"  reference_match_rms={analysis_obj.reference_match_rms}\n"
                 f"  need_limiter={analysis_obj.need_limiter}")

        # Save FIR data (conditional)
        save_fir_data(
            analysis_obj.mid_fir,
            sample_rate=44100,
            output_dir=output_dir,
            file_prefix=f"{target_basename}_{reference_basename}_Mid_FIR",
            save_plot=plot_fir
        )
        save_fir_data(
            analysis_obj.side_fir,
            sample_rate=44100,
            output_dir=output_dir,
            file_prefix=f"{target_basename}_{reference_basename}_Side_FIR",
            save_plot=plot_fir
        )

        # Save IR files
        save_fir_ir(
            analysis_obj.mid_fir,
            sample_rate=44100,
            output_path=os.path.join(output_dir, f"{target_basename}_{reference_basename}_Mid_FIR.wav"),
            normalize=False,
        )
        save_fir_ir(
            analysis_obj.side_fir,
            sample_rate=44100,
            output_path=os.path.join(output_dir, f"{target_basename}_{reference_basename}_Side_FIR.wav"),
            normalize=False,
        )

        # Now apply the matching process for each bit depth
        for bit_depth in bit_depths:
            if bit_depth == "float":
                suffix = "_32bit_float.aiff"
            elif bit_depth == 16:
                suffix = "_16bit.wav"
            elif bit_depth == 24:
                suffix = "_24bit.wav"
            else:
                continue  # Skip unknown bit depths

            out_file = os.path.join(output_dir, f"{target_basename}_{reference_basename}{suffix}")
            counter = 1
            # Avoid overwriting
            while os.path.exists(out_file):
                out_file = os.path.join(
                    output_dir, f"{target_basename}_{reference_basename}_{counter}{suffix}"
                )
                counter += 1

            log_info(f"Applying analysis => {out_file}")
            apply_analysis(
                input_file=target,
                analysis=analysis_obj,
                output_file=out_file,
                bit_depth=("float" if bit_depth == "float" else bit_depth)
            )

    log_info("Matchering completed!")

def process_stems_mode(
    target_file: str,
    reference_file: str,
    stems: list,
    bit_depths: list,
    output_dir: str,
    plot_fir: bool = False,
    analysis_file=None,
    reuse_analysis=False
):
    """
    1) Possibly load existing analysis or do a fresh one.
    2) Save FIR filter data.
    3) Apply analysis to the main target.
    4) Apply the same analysis to each stem.
    """
    os.makedirs(output_dir, exist_ok=True)

    target_stem = Path(target_file).stem
    reference_stem = Path(reference_file).stem

    if reuse_analysis and analysis_file and os.path.isfile(analysis_file):
        log_info(f"Loading analysis from '{analysis_file}'...")
        analysis_obj = load_analysis(analysis_file)
    else:
        # Step 1: Analyze
        log_info(f"Analyzing target='{target_file}' reference='{reference_file}' ...")
        analysis_obj = analyze(target_file, reference_file)
        # Always save new analysis
        if analysis_file:
            save_analysis(analysis_obj, analysis_file)
        else:
            default_json = os.path.join(output_dir, f"{target_stem}_{reference_stem}_analysis.json")
            save_analysis(analysis_obj, default_json)

    # Step 2: Save FIR filter data
    log_info("Saving FIR filter data and IR files...")
    save_fir_data(
        analysis_obj.mid_fir,
        sample_rate=44100,
        output_dir=output_dir,
        file_prefix="mid_fir",
        save_plot=plot_fir
    )
    save_fir_data(
        analysis_obj.side_fir,
        sample_rate=44100,
        output_dir=output_dir,
        file_prefix="side_fir",
        save_plot=plot_fir
    )

    save_fir_ir(
        analysis_obj.mid_fir,
        sample_rate=44100,
        output_path=os.path.join(output_dir, f"{target_stem}_{reference_stem}_Mid_FIR.wav"),
        normalize=False,
    )

    save_fir_ir(
        analysis_obj.side_fir,
        sample_rate=44100,
        output_path=os.path.join(output_dir, f"{target_stem}_{reference_stem}_Side_FIR.wav"),
        normalize=False,
    )

    # Step 3: Apply analysis to main target
    for bd in bit_depths:
        if bd == "float":
            suffix = "_32bit_float.aiff"
        elif bd == 16:
            suffix = "_16bit.wav"
        elif bd == 24:
            suffix = "_24bit.wav"
        else:
            continue

        out_file = os.path.join(output_dir, f"{target_stem}_{reference_stem}{suffix}")
        counter = 1
        while os.path.exists(out_file):
            out_file = os.path.join(output_dir, f"{target_stem}_{reference_stem}_{counter}{suffix}")
            counter += 1

        log_info(f"Applying analysis to main target => {out_file}")
        apply_analysis(
            input_file=target_file,
            analysis=analysis_obj,
            output_file=out_file,
            bit_depth=("float" if bd == "float" else bd)
        )

    # Step 4: Apply analysis to each stem
    for stem_path in stems:
        stem_basename = Path(stem_path).stem
        for bd in bit_depths:
            if bd == "float":
                suffix = "_32bit_float.aiff"
            elif bd == 16:
                suffix = "_16bit.wav"
            elif bd == 24:
                suffix = "_24bit.wav"
            else:
                continue

            out_file = os.path.join(output_dir, f"{stem_basename}_{reference_stem}{suffix}")
            counter = 1
            while os.path.exists(out_file):
                out_file = os.path.join(
                    output_dir, f"{stem_basename}_{reference_stem}_{counter}{suffix}"
                )
                counter += 1

            log_info(f"Applying analysis to stem '{stem_basename}' => {out_file}")
            apply_analysis(
                input_file=stem_path,
                analysis=analysis_obj,
                output_file=out_file,
                bit_depth=("float" if bd == "float" else bd)
            )

    log_info("Stems processing completed!")

# --------------------------------------------------------------------
# 5) Main Entry
# --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Matchering audio tracks.")
    parser.add_argument("target", help="Path to the target track or directory.")
    parser.add_argument("reference", help="Path to the reference track or directory.")
    parser.add_argument(
        "-b", "--bit-depths",
        help="Comma-separated bit depths to output (16, 24, float). Default: float",
        default="float"
    )
    parser.add_argument(
        "-o", "--output",
        help="Directory to save the matchered files. Default: Same folder as target track."
    )
    parser.add_argument(
        "-s", "--stems",
        help="Trigger stems mode (analysis + apply to stems). Provide comma-separated list of stem files or leave blank to be prompted.",
        nargs="?",
        const=""  # If user just does "-s", we get empty string in args.stems
    )
    parser.add_argument(
        "-p", "--plot-fir",
        help="Plot the FIR frequency responses after analysis (requires matplotlib).",
        action="store_true"
    )
    parser.add_argument(
        "-A", "--analysis-file",
        help="Path to analysis JSON file to load/save for reuse.",
        default=None
    )
    parser.add_argument(
        "-r", "--reuse-analysis",
        help="Reuse an existing analysis file (skip re-analysis). Must also specify --analysis-file.",
        action="store_true"
    )

    args = parser.parse_args()

    target_path = args.target.strip("'\"")
    reference_path = args.reference.strip("'\"")
    output_dir = args.output.strip("'\"") if args.output else None
    bit_depths = [bd.strip() for bd in args.bit_depths.split(",")]

    # Validate bit depths
    valid_depths = {"16", "24", "float"}
    invalid_depths = [bd for bd in bit_depths if bd not in valid_depths]
    if invalid_depths:
        log_warning(f"Invalid bit depths specified: {', '.join(invalid_depths)}")
        sys.exit(1)

    # Convert to int or leave as "float"
    refined_bit_depths = []
    for bd in bit_depths:
        if bd.isdigit():
            refined_bit_depths.append(int(bd))
        else:
            refined_bit_depths.append(bd)
    bit_depths = refined_bit_depths

    # Default output_dir is target's folder (if not specified)
    if not output_dir:
        output_dir = target_path if os.path.isdir(target_path) else os.path.dirname(target_path)
    os.makedirs(output_dir, exist_ok=True)

    # Handle stems mode
    if args.stems is not None:
        # Validate non-directory inputs
        if os.path.isdir(target_path) or os.path.isdir(reference_path):
            log_warning("Stems mode not possible if either target or reference is a directory.")
            sys.exit(1)

        # Parse stems
        if args.stems.strip():
            stems_list = [s.strip() for s in args.stems.split(",")]
        else:
            stems_input = input("Enter comma-separated paths to your stem files (or leave blank if none): ")
            stems_list = [s.strip() for s in stems_input.split(",")] if stems_input else []

        process_stems_mode(
            target_file=target_path,
            reference_file=reference_path,
            stems=stems_list,
            bit_depths=bit_depths,
            output_dir=output_dir,
            plot_fir=args.plot_fir,
            analysis_file=args.analysis_file,
            reuse_analysis=args.reuse_analysis
        )
        return

    # Otherwise, do normal directory-based or single-file processing
    if os.path.isdir(target_path):
        targets = find_audio_files(target_path)
    else:
        targets = [target_path]

    if os.path.isdir(reference_path):
        references = find_audio_files(reference_path)
    else:
        references = [reference_path]

    if not targets:
        log_warning(f"No audio files found in target path: {target_path}")
        sys.exit(1)
    if not references:
        log_warning(f"No audio files found in reference path: {reference_path}")
        sys.exit(1)

    # Generate combinations
    combinations = [(t, r) for t in targets for r in references]

    # Confirm combos if > 1
    if len(combinations) > 1:
        if not confirm_combinations(combinations):
            log_info("Aborting.")
            sys.exit(0)

    # Match them
    matcher_audio_combinations(
        combinations,
        output_dir,
        bit_depths,
        plot_fir=args.plot_fir,
        analysis_file=args.analysis_file,
        reuse_analysis=args.reuse_analysis
    )

if __name__ == "__main__":
    main()