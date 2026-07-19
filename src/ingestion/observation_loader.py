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

from src.utils.config import CONFIG, get_path


class ObservationLoader:
    """
    Load processed SoLEXS and HEL1OS observations.
    """

    def __init__(
        self,
        processed_directory: str | Path,
    ) -> None:

        self.processed_directory = Path(
            processed_directory
        ) if processed_directory is not None else get_path("processed")

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

        lightcurve_path = solexs_folder / CONFIG["files"]["observation_lightcurve"]
        if not lightcurve_path.exists():
            candidate = solexs_folder / Path(CONFIG["files"]["observation_lightcurve"]).name
            if candidate.exists():
                lightcurve_path = candidate

        lightcurve = pd.read_csv(lightcurve_path)

        ##################################################
        # HEL1OS
        ##################################################

        event_path = hel1os_folder / CONFIG["files"]["observation_event"]
        if not event_path.exists():
            candidate = hel1os_folder / Path(CONFIG["files"]["observation_event"]).name
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

    def load_all(
        self,
    ):

        solexs_root = (
            self.processed_directory
            / "solexs"
        )

        hel1os_root = (
            self.processed_directory
            / "hel1os"
        )

        solexs_folders = sorted(
            [
                folder

                for folder in solexs_root.iterdir()

                if folder.is_dir()

            ]
        )

        hel1os_folders = sorted(
            [
                folder

                for folder in hel1os_root.iterdir()

                if folder.is_dir()

            ]
        )

        for hel_folder in hel1os_folders:

            # Reuse first SoLEXS folder for now
            # until observation pairing is implemented.

            yield self.load(

                solexs_folders[0],

                hel_folder,

            )