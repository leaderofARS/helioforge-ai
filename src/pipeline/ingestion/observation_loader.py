"""
==========================================================
HELIO-FORGE (SOLAR PRELUDE)
observation_loader.py

Loads all observations.
==========================================================
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import PATH_CFG

# Observation-level filenames (these are fixed FITS-derived CSV names)
_LIGHTCURVE_FILENAME = "lightcurve.csv"
_EVENT_FILENAME      = "event.csv"


class ObservationLoader:
    """
    Load processed SoLEXS and HEL1OS observations.
    """

    def __init__(
        self,
        processed_directory: str | Path,
    ) -> None:

        self.processed_directory = (
            Path(processed_directory)
            if processed_directory is not None
            else PATH_CFG.preprocessing.processed
        )

    ##################################################
    # LOAD SINGLE OBSERVATION
    ##################################################

    def load(
        self,
        solexs_folder: str | Path,
        hel1os_folder: str | Path,
    ) -> dict[str, np.ndarray]:

        solexs_folder = Path(solexs_folder)
        hel1os_folder = Path(hel1os_folder)

        ##################################################
        # SOLEXS
        ##################################################

        lightcurve_path = solexs_folder / _LIGHTCURVE_FILENAME
        if not lightcurve_path.exists():
            candidate = solexs_folder / Path(_LIGHTCURVE_FILENAME).name
            if candidate.exists():
                lightcurve_path = candidate

        lightcurve = pd.read_csv(lightcurve_path)

        ##################################################
        # HEL1OS
        ##################################################

        event_path = hel1os_folder / _EVENT_FILENAME
        if not event_path.exists():
            candidate = hel1os_folder / Path(_EVENT_FILENAME).name
            if candidate.exists():
                event_path = candidate

        event = pd.read_csv(event_path)

        ##################################################
        # BUILD ARRAYS — SoLEXS (already 1 Hz)
        ##################################################

        timestamps  = lightcurve["TIME"].to_numpy(dtype=float)
        soft_signal = lightcurve["COUNTS"].to_numpy(dtype=float)

        ##################################################
        # BUILD ARRAYS — HEL1OS (photon events → 1 Hz)
        #
        # event.csv has one row per detected photon:
        #   mjd  : Modified Julian Date of detection
        #   ener : photon energy in keV
        #
        # SoLEXS TIME is in mission seconds (ISRO clock).
        # We cannot convert MJD ↔ mission seconds without the
        # exact epoch from the FITS header, so we align both
        # signals by RELATIVE time:
        #
        #   soft_rel[i] = TIME[i] - TIME[0]   (seconds from obs start)
        #   hard_rel[j] = (mjd[j] - mjd[0]) * 86400
        #
        # Then we bin photon energies into the same 1-second
        # integer bins as soft_rel, producing a 1-Hz hard signal
        # of identical length to soft_signal.
        ##################################################

        hard_signal = self._bin_events_to_1hz(
            event_mjd=event["mjd"].to_numpy(dtype=float),
            event_ener=event["ener"].to_numpy(dtype=float),
            soft_timestamps=timestamps,
        )

        ##################################################
        # FINAL RETURN — lengths guaranteed equal
        ##################################################

        n = len(timestamps)   # soft and hard are already matched

        return {
            "solexs_id":   solexs_folder.name,
            "hel1os_id":   hel1os_folder.name,
            "timestamps":  timestamps[:n],
            "soft_signal": soft_signal[:n],
            "hard_signal": hard_signal[:n],
        }

    # ------------------------------------------------------------------
    # HEL1OS event binning helper
    # ------------------------------------------------------------------

    @staticmethod
    def _bin_events_to_1hz(
        event_mjd: "np.ndarray",
        event_ener: "np.ndarray",
        soft_timestamps: "np.ndarray",
    ) -> "np.ndarray":
        """
        Bin HEL1OS photon events into the same 1-second time grid as
        the SoLEXS lightcurve.

        Strategy (epoch-free)
        ─────────────────────
        1. Convert both time axes to relative seconds from their own start:
               soft_rel[i] = TIME[i] - TIME[0]
               hard_rel[j] = (mjd[j] - mjd[0]) * 86400
        2. Compute the mean photon energy in each 1-second integer bin
           that aligns with soft_rel.  Bins with no events get 0.0.
        3. Return an array of the same length as soft_timestamps.

        Parameters
        ----------
        event_mjd       : (M,) photon detection times in MJD
        event_ener      : (M,) photon energies in keV
        soft_timestamps : (T,) SoLEXS TIME in mission seconds

        Returns
        -------
        hard_signal : (T,) mean keV per 1-second bin, aligned to soft_timestamps
        """
        T = len(soft_timestamps)

        if len(event_mjd) == 0:
            return np.zeros(T, dtype=np.float32)

        # Relative seconds
        soft_rel  = soft_timestamps - soft_timestamps[0]          # (T,)
        hard_secs = (event_mjd - event_mjd[0]) * 86400.0          # (M,)

        # Duration of the HEL1OS observation in relative seconds
        hard_duration = float(hard_secs[-1] - hard_secs[0]) if len(hard_secs) > 1 else 0.0
        soft_duration = float(soft_rel[-1]) if T > 1 else 0.0

        # Scale hard relative time to match soft duration
        # (compensates for any clock-rate differences between the two instruments)
        if hard_duration > 1.0 and soft_duration > 1.0:
            hard_secs = hard_secs * (soft_duration / hard_duration)

        # Bin photon energies into integer-second bins [0, T)
        bin_indices = np.floor(hard_secs).astype(np.int64)
        valid_mask  = (bin_indices >= 0) & (bin_indices < T)
        bin_indices = bin_indices[valid_mask]
        bin_ener    = event_ener[valid_mask]

        # Sum energy per bin then divide by count (mean keV/s)
        hard_sum   = np.bincount(bin_indices, weights=bin_ener,  minlength=T)
        hard_count = np.bincount(bin_indices,                    minlength=T).astype(float)
        hard_count[hard_count == 0] = 1.0        # avoid division by zero

        hard_signal = (hard_sum / hard_count).astype(np.float32)
        return hard_signal



    ##################################################
    # LOAD ALL OBSERVATIONS
    ##################################################

    def load_all(self):
        """
        Yield all paired (SoLEXS, HEL1OS) observations.

        Pairing strategy — date-based (fixes the documented known limitation)
        ─────────────────────────────────────────────────────────────────────
        Both instruments encode the observation date in the folder name:
          SoLEXS : AL1_SLX_L1_YYYYMMDD_v*.0   → date token at position 3
          HEL1OS : HLS_YYYYMMDD_HHMMSS_*sec_*  → date token at position 1

        We extract the 8-digit date from each folder name, convert to int,
        then for each SoLEXS observation find the closest HEL1OS observation
        within MAX_DATE_GAP_DAYS days.  If no HEL1OS match is within that
        window, the SoLEXS observation is skipped with a warning.

        This avoids the old bug where alphabetically sorted index-0 SoLEXS
        (2024-03-05) was paired with index-0 HEL1OS (2023-12-23), producing
        a near-zero length signal after length truncation.
        """
        import re

        MAX_DATE_GAP_DAYS = 7      # observations within 7 days are considered a valid pair

        solexs_root = self.processed_directory / "solexs"
        hel1os_root = self.processed_directory / "hel1os"

        if not solexs_root.exists() or not hel1os_root.exists():
            raise FileNotFoundError(
                f"Processed observation folders missing under {self.processed_directory}.\n"
                f"Expected: {solexs_root} and {hel1os_root}\n"
                "Please run Stage 1 preprocessing (preprocess.py) first."
            )

        solexs_folders = sorted([f for f in solexs_root.iterdir() if f.is_dir()])
        hel1os_folders = sorted([f for f in hel1os_root.iterdir() if f.is_dir()])

        if not solexs_folders or not hel1os_folders:
            raise RuntimeError(
                f"No processed observation directories found in {solexs_root} or {hel1os_root}.\n"
                "Please run Stage 1 preprocessing (preprocess.py) first."
            )

        # ── Extract 8-digit date from folder names ────────────────────────
        def _extract_date(folder: Path) -> int | None:
            """Return YYYYMMDD as int, or None if not found."""
            m = re.search(r"(\d{8})", folder.name)
            return int(m.group(1)) if m else None

        # Build lookup: date_int → hel1os_folder (keep first match per date)
        hel1os_by_date: dict[int, Path] = {}
        for h in hel1os_folders:
            d = _extract_date(h)
            if d is not None and d not in hel1os_by_date:
                hel1os_by_date[d] = h

        hel1os_dates = sorted(hel1os_by_date.keys())

        def _nearest_hel1os(solexs_date: int) -> Path | None:
            """Return the closest HEL1OS folder within MAX_DATE_GAP_DAYS."""
            if not hel1os_dates:
                return None
            # Convert YYYYMMDD ints to a rough day difference (not calendar-exact
            # but accurate enough for a 7-day window within the same month/year)
            def _to_days(d: int) -> int:
                y, m, day = d // 10000, (d % 10000) // 100, d % 100
                return y * 365 + m * 30 + day       # approximate, good enough

            s_days = _to_days(solexs_date)
            best_date = min(hel1os_dates, key=lambda d: abs(_to_days(d) - s_days))
            if abs(_to_days(best_date) - s_days) <= MAX_DATE_GAP_DAYS:
                return hel1os_by_date[best_date]
            return None

        # ── Yield paired observations ─────────────────────────────────────
        paired = 0
        skipped_no_match = 0

        for slx in solexs_folders:
            slx_date = _extract_date(slx)
            if slx_date is None:
                print(f"[ObservationLoader] Cannot parse date from: {slx.name} — skipping")
                skipped_no_match += 1
                continue

            hel = _nearest_hel1os(slx_date)
            if hel is None:
                print(
                    f"[ObservationLoader] No HEL1OS match within {MAX_DATE_GAP_DAYS} days "
                    f"for SoLEXS {slx.name} (date={slx_date}) — skipping"
                )
                skipped_no_match += 1
                continue

            try:
                obs = self.load(slx, hel)
                if len(obs["soft_signal"]) == 0 or len(obs["hard_signal"]) == 0:
                    print(
                        f"[ObservationLoader] Empty signals for pair "
                        f"{obs['solexs_id']} / {obs['hel1os_id']} — skipping"
                    )
                    continue
                paired += 1
                yield obs
            except Exception as exc:
                print(
                    f"[ObservationLoader] Error loading pair "
                    f"({slx.name} / {hel.name}): {exc}"
                )
                continue

        print(
            f"[ObservationLoader] Pairing complete — "
            f"yielded {paired} pairs, skipped {skipped_no_match} (no date match)."
        )