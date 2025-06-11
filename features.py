# Feature extraction functions
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import pywt


SENSOR_COLS = ["Acc_X", "Acc_Y", "Acc_Z", "Gyro_X", "Gyro_Y", "Gyro_Z"]

# Default colours for each sensor when visualising data.  Using a fixed
# mapping ensures the check boxes and plot lines share the same colour
# cues across the application.
SENSOR_COLORS = {
    "Acc_X": "#e6194b",  # red
    "Acc_Y": "#3cb44b",  # green
    "Acc_Z": "#4363d8",  # blue
    "Gyro_X": "#f58231",  # orange
    "Gyro_Y": "#911eb4",  # purple
    "Gyro_Z": "#46f0f0",  # cyan
}


def time_domain_features(df_window: pd.DataFrame) -> dict:
    """Compute basic time-domain statistics for each sensor column."""
    features = {}
    for col in SENSOR_COLS:
        x = df_window[col].values
        features.update(
            {
                f"{col}_mean": np.mean(x),
                f"{col}_std": np.std(x),
                f"{col}_min": np.min(x),
                f"{col}_max": np.max(x),
                f"{col}_ptp": np.max(x) - np.min(x),
                f"{col}_rms": np.sqrt(np.mean(x ** 2)),
                f"{col}_skew": skew(x),
                f"{col}_kurtosis": kurtosis(x),
            }
        )
    return features


def frequency_domain_features(df_window: pd.DataFrame, Fs: int = 100) -> dict:
    """Compute frequency-domain features using the FFT."""
    features = {}
    N = len(df_window)
    for col in SENSOR_COLS:
        signal = df_window[col].values
        fft_vals = np.fft.rfft(signal)
        mag = np.abs(fft_vals)
        freqs = np.fft.rfftfreq(n=N, d=1.0 / Fs)

        energy = np.sum(mag ** 2)
        if np.sum(mag) == 0:
            centroid = 0.0
            bandwidth = 0.0
        else:
            centroid = np.sum(freqs * mag) / np.sum(mag)
            bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * mag) / np.sum(mag))
        peak_freq = freqs[np.argmax(mag)]

        features.update(
            {
                f"{col}_spec_energy": energy,
                f"{col}_spec_centroid": centroid,
                f"{col}_spec_bw": bandwidth,
                f"{col}_spec_peakfreq": peak_freq,
            }
        )
    return features


def wavelet_features(df_window: pd.DataFrame, wavelet: str = "db1", level: int = 2) -> dict:
    """Compute simple wavelet energy and entropy features."""
    features = {}
    for col in SENSOR_COLS:
        signal = df_window[col].values
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        all_coeffs = np.concatenate(coeffs)
        wave_energy = np.sum(all_coeffs ** 2)
        abs_coeffs = np.abs(all_coeffs)
        total_sum = np.sum(abs_coeffs)
        if total_sum == 0:
            wave_entropy = 0.0
        else:
            p = abs_coeffs / total_sum
            wave_entropy = -np.sum(p * np.log2(p + 1e-12))
        features.update({f"{col}_wave_energy": wave_energy, f"{col}_wave_entropy": wave_entropy})
    return features


def extract_features(df: pd.DataFrame, feature_list: list) -> pd.DataFrame:
    """Aggregate selected features for each file in ``df``.

    When a ``folder`` column is present it is kept as part of the grouping so
    that predictions can later be mapped back to their original location.
    """

    if not feature_list:
        return df

    group_cols = ["source_file"]
    if "folder" in df.columns:
        group_cols.insert(0, "folder")

    rows = []
    for keys, group in df.groupby(group_cols):
        row = {}
        if "time_domain" in feature_list:
            row.update(time_domain_features(group))
        if "freq_domain" in feature_list:
            row.update(frequency_domain_features(group))
        if "wavelet" in feature_list:
            row.update(wavelet_features(group))

        if isinstance(keys, tuple):
            # ``pandas`` 2.x always returns tuples when ``group_cols`` is a list
            # even if it only contains a single column.  Handle both the
            # two-value case ``(folder, source_file)`` and the single value
            # ``(source_file,)`` that older versions returned as a scalar.
            if len(keys) == 2:
                row["folder"], row["source_file"] = keys
            elif len(keys) == 1:
                row["source_file"] = keys[0]
            else:
                # Unexpected length, fallback to last element as the file name
                row["source_file"] = keys[-1]
                if len(keys) > 1:
                    row["folder"] = keys[0]
        else:
            row["source_file"] = keys
        if "label" in group.columns:
            row["label"] = group["label"].iloc[0]
        rows.append(row)

    return pd.DataFrame(rows)
