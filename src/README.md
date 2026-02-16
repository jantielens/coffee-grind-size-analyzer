# Coffee Grind Size Analyser — Usage

## Prerequisites

- Python 3.10+
- A printed [reference sheet](../reference-sheet.pdf) (A4, print at 100% scale)

## Installation

```bash
pip install -r requirements.txt
```

> **Note:** Cellpose and PyTorch are listed in requirements.txt for comparison purposes.
> If you only need the watershed pipeline (recommended), you can skip them —
> the script will fall back gracefully.

## Source Files

| File | Description |
|------|-------------|
| `analyze.py` | CLI entry point & pipeline orchestration (batch processing, result aggregation) |
| `constants.py` | Shared constants, geometry helpers, and grind-setting calibration curve |
| `detection.py` | ArUco marker detection with geometric fallback |
| `preprocessing.py` | Perspective warp, marker masking, white-balance normalisation |
| `segmentation.py` | Watershed (primary) and Cellpose (optional) particle segmentation |
| `measurement.py` | Particle measurement — size, shape, filtering, and distribution statistics (D10/D50/D90, span, fines, boulders, D[4,3], D[3,2]) |
| `visualization.py` | Overlays, distribution plots, pipeline summary PNGs, correlation charts |
| `generate-reference-sheet.py` | Generates the printable A4 reference sheet PDF |

## Preparing Your Photos

1. Print the reference sheet on plain white paper
2. Spread a thin, even layer of coffee grounds on the white area between the four markers
3. Take a photo with your phone's **flash on**, from roughly 20–30 cm above
4. Make sure all four ArUco markers are clearly visible

## Running the Analysis

### Report mode (unlabelled photos)

Don't know the grind setting? Use `--report` to have it estimated automatically:

```bash
python analyze.py photo.jpg --report
python analyze.py photos/ --report --output report-results/
```

For each photo the script produces `summary.png` and estimates the DF54 grind
setting from the measured median particle diameter.  A `report.csv` with all
estimates is written to the output directory.

### Single image

```bash
python analyze.py photo.jpg --grind-setting 40
```

### Batch (entire folder)

```bash
python analyze.py photos/ --output results/
```

The grind setting is inferred from the filename (e.g., `setting40-1.jpg` → setting 40).
Use `--grind-setting` to override when processing a single image.

### Command-Line Options

| Flag | Description | Default |
|------|-------------|---------|
| `-o`, `--output` | Output directory | `results/` |
| `-m`, `--model` | Model to run: `watershed`, `cellpose`, `both` | `watershed` |
| `-g`, `--grind-setting` | Override grind setting for all images | auto from filename |
| `--expected-diam` | Expected particle diameter in mm (Cellpose only) | `0.7` || `--report` | Report mode: estimate grind setting from unlabelled photos | off |
## Output

For each image, the script produces:

| File | Description |
|------|-------------|
| `warped.jpg` | Perspective-corrected top-down view |
| `normalized.jpg` | White-balance corrected image |
| `overlay_watershed.jpg` | Segmentation overlay with detected particles |
| `distribution_watershed.png` | Particle size distribution histogram |
| `particles_watershed.csv` | Per-particle measurements (area, diameter, axes, shape) |
| `summary.png` | Full pipeline visualisation — histogram, visual gauges, distribution statistics, all processing steps |

When processing a batch, a `summary.csv` and `correlation.png` are also generated in the output root.

In report mode, a `report.csv` is generated with estimated DF54 settings for each image.

## Generating the Reference Sheet

```bash
python generate-reference-sheet.py
```

This produces `reference_sheet.pdf` — an A4 page with four ArUco markers at known positions. Print at 100% scale (no "fit to page").

## Example Output

See the [examples/](../examples/) folder for sample output from a setting-40 grind.
