# Dataset

The dataset is stored as line-preserving JSONL parts under `datasets_pc_parts/` and `datasets_pc_sft_parts/`. Run:

```bash
python scripts/materialize_datasets.py
```

to recreate the `datasets_pc/` and `datasets_pc_sft/` directories used by the training and evaluation scripts.
