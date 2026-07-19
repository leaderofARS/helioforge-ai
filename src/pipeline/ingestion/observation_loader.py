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
        # BUILD ARRAYS
        ##################################################

        timestamps = lightcurve[
            "TIME"
        ].to_numpy(
            dtype=float
        )

        soft_signal = lightcurve[
            "COUNTS"
        ].to_numpy(
            dtype=float
        )

        # Temporary hard signal
        hard_signal = event[
            "ener"
        ].to_numpy(
            dtype=float
        )

        ##################################################
        # MATCH LENGTHS
        ##################################################

        n = min(

            len(timestamps),

            len(soft_signal),

            len(hard_signal),

        )

        return {
            "solexs_id": solexs_folder.name,
            "hel1os_id": hel1os_folder.name,

            "timestamps": timestamps[:n],

            "soft_signal": soft_signal[:n],

            "hard_signal": hard_signal[:n],

        }

    ##################################################
    # LOAD ALL OBSERVATIONS
    ##################################################

    def load_all(self):
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

        # Pair observations by index up to available count
        n_pairs = min(len(solexs_folders), len(hel1os_folders))
        for i in range(n_pairs):
            try:
                obs = self.load(solexs_folders[i], hel1os_folders[i])
                if len(obs["soft_signal"]) == 0 or len(obs["hard_signal"]) == 0:
                    print(
                        f"[ObservationLoader] Skipping empty observation pair: "
                        f"{obs['solexs_id']} / {obs['hel1os_id']}"
                    )
                    continue
                yield obs
            except Exception as exc:
                print(
                    f"[ObservationLoader] Error loading pair "
                    f"({solexs_folders[i].name} / {hel1os_folders[i].name}): {exc}"
                )
                continue