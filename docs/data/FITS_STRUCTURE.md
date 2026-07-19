# HELIO-FORGE (SOLAR PRELUDE) FITS Structure Guide

## 1. Project Overview

HELIO-FORGE (SOLAR PRELUDE) is a scientific data preprocessing and engineering workflow for the Aditya-L1 mission payload data from two instruments:

- HEL1OS: High Energy L1 Orbiting X-ray Spectrometer
- SoLEXS: Solar Low Energy X-ray Spectrometer

The repository focuses on inspecting raw Level-1 science products, understanding their FITS structure, validating data quality, extracting metadata, and preparing files for downstream scientific analysis and preprocessing.

This documentation is based on the repository layout, the raw data directories, the metadata CSV files, and the preprocessing notebook in the notebooks folder.

---

## 2. Repository Folder Structure

```text
HELIO-FORGE AI DATA ENGINEERING/
├── README.md
├── Helio-forge/
│   ├── data/
│   │   ├── metadata.csv
│   │   ├── solexs_metadata.csv
│   │   └── raw/
│   │       ├── hel1os/
│   │       └── solexs/
│   ├── docs/
│   │   └── FITS_STRUCTURE.md
│   ├── notebooks/
│   │   └── solar_prelude_preprocessing.ipynb
│   ├── preprocessing/
│   ├── processed/
│   └── reports/
```

### Purpose of Each Folder

- data/: Contains raw scientific products and exported metadata summaries.
- docs/: Stores project documentation, including this FITS structure guide.
- notebooks/: Contains the main exploratory and preprocessing workflow notebook.
- preprocessing/: Intended for preprocessing modules and data handling logic.
- processed/: Expected destination for cleaned or derived scientific products.
- reports/: Intended for generated validation or analysis reports.

---

## 3. HEL1OS Dataset Structure

The HEL1OS raw data is organized under:

```text
Helio-forge/data/raw/hel1os/
└── HLS_20240624_012528_38058sec_lev1_V111/
    └── 2024/06/24/HLS_20240624_012528_38058sec_lev1_V111/
        ├── aux_data/aux_backup/
        ├── cdte/
        ├── czt/
        └── events/
```

### HEL1OS File Categories

| Category | Typical Files | Purpose |
|---|---|---|
| Event data | evt.fits | Raw photon event lists |
| Housekeeping | hk.fits | Instrument health and telemetry |
| Good Time Intervals | gticdte1.fits, gticdte2.fits, gticzt1.fits, gticzt2.fits | Valid observation intervals |
| Spectra | hel1os_*_spectra_*.fits | Time-resolved spectra |
| Lightcurves | lightcurve_*.fits | Count-rate products in energy bands |

### HEL1OS Scientific Files

- evt.fits: Primary event list containing individual photon detections.
- hk.fits: Housekeeping telemetry for temperatures, voltages, and detector health metrics.
- gticdte*.fits and gticzt*.fits: GTI tables for detector-specific valid times.
- hel1os_cdte_spectra_*.fits: CdTe spectra products.
- hel1os_czt_spectra_*.fits: CZT spectra products.
- lightcurve_*.fits: Multi-band lightcurves for detector modules.

---

## 4. SoLEXS Dataset Structure

The SoLEXS dataset is organized under:

```text
Helio-forge/data/raw/solexs/
└── AL1_SLX_L1_20240403_v1.0/
    └── AL1_SLX_L1_20240403_v1.0/
        ├── SDD1/
        └── SDD2/
```

### SoLEXS File Categories

| Category | Typical Files | Purpose |
|---|---|---|
| Good Time Intervals | *.gti | Observation intervals for valid science time |
| Lightcurve | *.lc | Lightcurve product |
| Spectrum | *.pi | Energy-channel spectrum product |

### SoLEXS Scientific Files

- AL1_SOLEXS_20240403_SDD1_L1.gti: GTI file for SDD1.
- AL1_SOLEXS_20240403_SDD2_L1.gti: GTI file for SDD2.
- AL1_SOLEXS_20240403_SDD2_L1.lc: SoLEXS lightcurve file.
- AL1_SOLEXS_20240403_SDD2_L1.pi: SoLEXS spectrum file in FITS binary table form.

---

## 5. FITS File Types Used in the Project

The project primarily uses FITS files and related science products with the following roles:

| File Type | Examples | Description |
|---|---|---|
| Event list | evt.fits | Contains individual photon events |
| Housekeeping | hk.fits | Instrument state and health parameters |
| GTI table | *.gti, gticdte*.fits, gticzt*.fits | Defines valid time intervals |
| Spectra | *.fits, *.pi | Time-binned spectral products |
| Lightcurve | *.fits, *.lc | Time-binned count-rate products |
| Metadata export | metadata.csv, solexs_metadata.csv | Structured summaries of scientific files |

---

## 6. HDU Structure

FITS files in this project generally contain one or more Header Data Units (HDUs). In the notebook workflow, the files were inspected using Astropy to determine the available HDUs and table contents.

### Common HDU Patterns

- Primary HDU: Contains file-level metadata and header keywords.
- Binary table HDU: Stores tabular scientific data such as events, GTIs, or spectra.
- Multiple HDUs: Some HEL1OS lightcurve files contain several extensions representing different energy bands.

### Observed Structure in the Notebook

- HEL1OS evt.fits: Several event-table HDUs corresponding to detector modules.
- HEL1OS housekeeping and GTI files: Compact FITS tables with one data extension.
- SoLEXS GTI, lightcurve, and spectrum files: FITS files with a primary header and a science data table extension.

> The exact HDU names and extension layout vary by file, and the notebook was used to discover them rather than assume a fixed schema.

---

## 7. Important FITS Headers

The preprocessing notebook extracted metadata from the FITS primary headers using the following common keywords:

| Header Keyword | Meaning |
|---|---|
| INSTRUME | Instrument name, such as HEL1OS or SoLEXS |
| TELESCOP | Telescope or mission identifier |
| DATE-OBS | Observation date |
| OBJECT | Target or object name |
| CREATOR | Processing pipeline or creator name |

### Observed Metadata Values

From the repository metadata files:

- Instrument names: HEL1OS and SoLEXS
- Telescope: Aditya-L1
- Creator: HEL1OS-L1-PIPELINE for HEL1OS products
- SoLEXS metadata indicates the processing pipeline was identified as solexs_pipeline-1.4 in the notebook workflow

---

## 8. Scientific Files Description

### 8.1 HEL1OS Event Files

The raw event file contains photon detections tagged with detector and timing information. In the notebook, the event FITS file was inspected as the main source for scientific event analysis.

Typical information present in event tables includes:

- timestamp values
- detector-specific timing or onboard clock values
- detector temperature information
- energy/channel information
- detector module identifiers

### 8.2 HEL1OS Housekeeping Files

Housekeeping files provide operational telemetry used to assess instrument health. These files are important for filtering out periods when the detector behavior may be unstable.

### 8.3 HEL1OS GTI Files

GTI files define valid time intervals for scientific analysis. They are used to limit downstream processing to intervals where the instrument data is considered reliable.

### 8.4 SoLEXS GTI Files

SoLEXS GTI files contain valid observation intervals. The notebook reported that the GTI file for SDD2 contained three valid intervals with no missing values.

### 8.5 SoLEXS Lightcurve Files

SoLEXS lightcurve files store temporally binned count-rate information. These are useful for inspecting flux evolution over time.

### 8.6 SoLEXS Spectrum Files

The SoLEXS spectrum files are more complex than a simple tabular table. The notebook observed that the .pi product stores multidimensional information where the CHANNEL and COUNTS columns contain arrays of values for each spectrum record.

---

## 9. Data Flow

The repository workflow follows a logical progression from raw mission data to scientifically prepared products:

```text
Raw mission FITS files
    -> Inspect FITS structure and HDUs
    -> Extract file-level metadata
    -> Validate scientific files
    -> Apply GTI filtering and quality checks
    -> Prepare lightcurves, spectra, and derived products
```

### Main Processing Flow

1. Locate raw science files in the data/raw hierarchy.
2. Open each FITS file with Astropy.
3. Inspect HDU layout and column structure.
4. Extract metadata from primary headers.
5. Validate completeness and consistency of files.
6. Apply timing-based filtering using GTI intervals when required.
7. Prepare data products for scientific analysis.

---

## 10. Preprocessing Workflow

The notebook describes a multi-phase workflow that can be summarized as follows:

### Phase 1 — Dataset Exploration

- Locate raw FITS files automatically.
- Inspect available files and directory structure.
- Identify the relevant event, housekeeping, GTI, and spectral products.

### Phase 2 — Dataset Profiling

- Inspect dataset shape, columns, and data types.
- Review whether data tables are scalar or array-based.
- Check for missing values and structural consistency.

### Phase 3 — Metadata Extraction

- Read header keywords from the FITS files.
- Extract instrument, telescope, observation date, object, and creator metadata.
- Store the results in CSV files for reference.

### Phase 4 — Dataset Validation

- Verify that the expected scientific files are present.
- Confirm that metadata extraction completed successfully.
- Check for missing values and unexpected data structure differences.

### Phase 5 — HEL1OS Scientific Preprocessing

- Explore event data.
- Inspect GTI files.
- Review housekeeping data.
- Perform timestamp synchronization and data-quality checks.

### Phase 6 — SoLEXS Exploration and Scientific Preprocessing

- Explore GTI, lightcurve, and spectrum products.
- Profile lightcurve datasets.
- Inspect spectrum structure.
- Extract metadata and validate the SoLEXS dataset.
- Prepare scientific products for further analysis.

---

## 11. Metadata Extraction

Metadata extraction plays an important role in this project. The notebook reads metadata from the FITS primary headers and stores it into structured CSV files.

### Metadata Captured

- filename
- file type
- file size
- number of HDUs
- instrument name
- telescope name
- observation date
- object name
- creator name

### Output Files

- data/metadata.csv for HEL1OS-related products
- data/solexs_metadata.csv for SoLEXS products

These files are useful for tracking the dataset inventory and supporting reproducible scientific processing.

---

## 12. Dataset Validation

Validation in this workflow includes checking that:

- all expected files are present
- each FITS file opens successfully
- metadata extraction completes without missing critical values
- the GTI tables are populated with valid intervals
- spectrum files contain the expected multidimensional columns

The notebook explicitly reports successful validation for the SoLEXS scientific files and notes that every file was processed without errors.

---

## 13. Timestamp Synchronization

One of the documented preprocessing tasks is timestamp synchronization. The notebook includes a dedicated step for aligning the timing information between instrument products and scientific observation intervals.

### Why This Matters

Scientific preprocessing depends on consistent time references so that:

- event data can be filtered by valid intervals
- lightcurves can be aligned with observation windows
- spectral integration windows can be interpreted correctly

### Practical Note

The project uses FITS timing information and GTI intervals as the main reference for temporal filtering. The exact conversion method depends on the file type and the time units stored in each product.

---

## 14. Scientific Preprocessing

Scientific preprocessing in this project is centered on preparing instrument data for analysis while preserving the integrity of the observation windows.

### Typical Preprocessing Steps

- inspect raw FITS files and identify their scientific tables
- validate GTI intervals and select valid time windows
- review housekeeping telemetry for instrument state consistency
- filter or clean event data according to data-quality rules
- generate or prepare lightcurves and spectra for downstream analysis

### Notes on Scientific Products

- Event data is the most fundamental photon-level data source.
- Lightcurves are useful for visualizing temporal variability.
- Spectra are important for energy-resolved analysis.
- GTI tables are essential for avoiding invalid observation periods.

---

## 15. Final Dataset Outputs

The intended outputs of the workflow are scientific datasets and derived products that are better structured for analysis. These may include:

- cleaned or filtered event lists
- validated GTI-based time selections
- derived lightcurve products
- spectral datasets prepared for analysis
- metadata summaries in CSV form

The repository currently contains the raw science files plus metadata summaries, while the processed outputs are expected to be generated as preprocessing workflows advance.

---

## 16. Challenges Encountered

The project documentation and notebook indicate several practical challenges that are common in space-science preprocessing:

- FITS files can contain multiple HDUs with different scientific tables.
- Some products, especially spectra, store array-valued columns rather than simple scalar columns.
- Time synchronization is essential for correctly combining data from different products.
- Data quality depends not only on the event content but also on valid observation windows and instrument health conditions.
- The exact schema of each file must be discovered from the data rather than assumed from a single universal template.

---

## 17. Notes for Developers

- Use the notebook as the primary reference for the implemented exploration workflow.
- Treat the raw FITS files as the source of truth for scientific structure.
- When inspecting new files, verify the HDU layout before assuming column names or data types.
- Preserve metadata extraction as a regular part of the workflow.
- Keep validation steps explicit so that preprocessing remains reproducible.
- If new product types are added, document their HDU structure and header keywords in the same style.

---

## 18. Summary

HELIO-FORGE (SOLAR PRELUDE) is a mission-oriented preprocessing workflow for HEL1OS and SoLEXS data. The core scientific challenge is not only to read the FITS files, but also to understand their structure, validate their integrity, synchronize timestamps, and prepare scientifically meaningful products. The repository’s notebook and metadata files form the practical basis for this documentation.
