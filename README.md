# RADOLAN gap infilling

This repository contains a Jupyter notebook that trains a U-Net to reconstruct
missing values in RADOLAN precipitation fields.

## Setup

```bash
uv sync --locked --group dev
uv run python -m ipykernel install --user --name hackfill --display-name "Python (hackfill)"
```

Open `infill_radolan.ipynb` and select the **Python (hackfill)** kernel. Update
the data, observation and mask paths in the notebook.

The RADOLAN data, model checkpoints, logs, and local data paths are not included.
