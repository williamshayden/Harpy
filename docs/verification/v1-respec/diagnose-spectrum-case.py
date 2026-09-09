"""One preselected failed classical episode; frozen source, no adaptation or models."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from harpy.analysis import analyze
from harpy.envs.robustness import CLEAN_CONDITION, ROBUSTNESS_CONDITIONS, RobustnessEvidenceCache
from harpy.envs.spectrum import GYM_ANALYSIS_CONFIG, LOG_FREQUENCY_GRID_HZ

ROOT = Path(__file__).resolve().parent
report_bytes = (ROOT / "robustness.json").read_bytes()
report = json.loads(report_bytes)
# Fixed rule: first failed Spectrum Peak record under noise-10db in report order.
record = next(
    row for row in report["records"]
    if row["actor_name"] == "spectrum-peak"
    and row["condition_id"] == "noise-10db"
    and not row["terminal"]["submitted_success"]
)
episode = next(row for row in report["episodes"] if row["id"] == record["episode_id"])
condition = next(item for item in ROBUSTNESS_CONDITIONS if item.id == record["condition_id"])
source = episode["source_pitch_cents"]
seed = episode["nuisance_seed"]
cache = RobustnessEvidenceCache(max_bytes=0)
noisy = cache.evidence(source, condition, seed)
clean = cache.evidence(source, CLEAN_CONDITION, seed)
assert noisy.waveform_sha256 == record["initial_evidence"]["waveform_sha256"]
assert noisy.spectrum_sha256 == record["initial_evidence"]["spectrum_sha256"]
assert abs(noisy.realized_snr_db - record["initial_evidence"]["realized_snr_db"]) < 1e-12

fs, size = 48_000, 262_144
window = np.hanning(size)
freq = np.fft.rfftfreq(size, 1 / fs)
coordinates = np.arange(1100, 10901, 5)
feasible = np.flatnonzero((coordinates >= 4800) & (coordinates <= 7200))
noise = noisy.waveform.astype(np.float64) - clean.waveform.astype(np.float64)

def spectrum(waveform):
    amplitude = 2 * np.abs(np.fft.rfft(np.asarray(waveform, dtype=np.float64) * window)) / window.sum()
    amplitude[-1] *= 0.5
    with np.errstate(divide="ignore"):
        levels = np.maximum(20 * np.log10(amplitude), -120)
    return amplitude, levels

def cents(hz):
    return 6900 + 1200 * math.log2(float(hz) / 440)

amplitude, db = spectrum(noisy.waveform)
clean_amplitude, clean_db = spectrum(clean.waveform)
noise_amplitude, noise_db = spectrum(noise)
linear_index = 1 + int(np.argmax(amplitude[1:]))
analysis = analyze(noisy.waveform, fs, GYM_ANALYSIS_CONFIG)
public_index = int(feasible[np.argmax(noisy.spectrum[feasible])])
clean_index = int(feasible[np.argmax(clean.spectrum[feasible])])
assert float(coordinates[public_index]) == record["estimates"][0]["estimated_candidate_cents"]
public_db = np.interp(LOG_FREQUENCY_GRID_HZ, freq[1:], db[1:])
clean_public_db = np.interp(LOG_FREQUENCY_GRID_HZ, freq[1:], clean_db[1:])
noise_public_db = np.interp(LOG_FREQUENCY_GRID_HZ, freq[1:], noise_db[1:])
np.testing.assert_array_equal(((np.clip(public_db, -120, 0) + 120) / 120).astype(np.float32), noisy.spectrum)
true_hz = 440 * 2 ** ((source - 6900) / 1200)
nearest = int(np.argmin(np.abs(coordinates - source)))

def grid_row(index):
    return {
        "cents": int(coordinates[index]),
        "frequency_hz": float(LOG_FREQUENCY_GRID_HZ[index]),
        "offset_from_true_hz": float(LOG_FREQUENCY_GRID_HZ[index] - true_hz),
        "offset_in_fft_bins": float((LOG_FREQUENCY_GRID_HZ[index] - true_hz) / (fs / size)),
        "noisy_level_dbfs": float(public_db[index]),
        "clean_level_dbfs": float(clean_public_db[index]),
        "noise_component_level_dbfs": float(noise_public_db[index]),
        "public_normalized_value": float(noisy.spectrum[index]),
    }

summary = {
    "selection": "First failed Spectrum Peak noise-10db episode in frozen report order; one case, no model, no adaptation.",
    "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
    "episode": episode,
    "condition": condition.to_document(),
    "record": record,
    "verified_original_waveform_and_spectrum_hashes": True,
    "fft_frames": size,
    "fft_bin_width_hz": fs / size,
    "hann_main_lobe_first_null_offset_hz_approx": 2 * fs / size,
    "true_frequency_hz": true_hz,
    "linear_fft_peak": {
        "frequency_hz": float(freq[linear_index]),
        "cents": cents(freq[linear_index]),
        "error_cents": cents(freq[linear_index]) - source,
        "level_dbfs": float(db[linear_index]),
    },
    "analysis_quadratic_peak": {
        "frequency_hz": analysis.peak_frequency_hz,
        "cents": cents(analysis.peak_frequency_hz),
        "error_cents": cents(analysis.peak_frequency_hz) - source,
        "level_dbfs": analysis.peak_level_dbfs,
    },
    "public_feasible_peak": grid_row(public_index),
    "clean_public_feasible_peak": grid_row(clean_index),
    "nearest_grid_point": grid_row(nearest),
    "grid_neighborhood": [grid_row(i) for i in range(nearest - 2, nearest + 3)],
    "top_five_public_feasible_bins": [grid_row(int(i)) for i in feasible[np.argsort(noisy.spectrum[feasible])[-5:][::-1]]],
    "same_episode_all_actor_records": [
        {"actor": r["actor_name"], "initial_estimates": r["estimates"], "terminal": r["terminal"]}
        for r in report["records"] if r["episode_id"] == episode["id"] and r["condition_id"] == condition.id
    ],
    "scope": "Explains a single selected observed failure. Does not estimate prevalence, alter the frozen encoder, or measure an alternative cohort.",
}
(ROOT / "spectrum-failure-case.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
with (ROOT / "spectrum-failure-case-grid.csv").open("w", newline="") as handle:
    rows = [grid_row(int(i)) for i in feasible]
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
with (ROOT / "spectrum-failure-case-linear.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["frequency_hz", "offset_from_true_hz", "noisy_level_dbfs", "clean_level_dbfs", "noise_level_dbfs"])
    for i in np.flatnonzero(np.abs(freq - true_hz) <= 3):
        writer.writerow([float(freq[i]), float(freq[i] - true_hz), float(db[i]), float(clean_db[i]), float(noise_db[i])])
print(json.dumps({key: value for key, value in summary.items() if key not in {"record", "same_episode_all_actor_records"}}, indent=2))
