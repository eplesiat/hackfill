# RADOLAN gap infilling

This repository contains a Jupyter notebook that trains a U-Net to reconstruct
missing values in RADOLAN precipitation fields.

## Setup

```bash
uv sync --locked --group dev
uv run python -m ipykernel install --user --name hackfill --display-name "Python (hackfill)"
```

Open `infill_radolan.ipynb` and select the **Python (hackfill)** kernel. Update
the observation path in the notebook and list the reference NetCDF files, one
path per line, in `data_files.txt`.

The RADOLAN data, model checkpoints, logs, and local data paths are not included.
