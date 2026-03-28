# Split Repo ML on Snowflake — ML Training

Model training, promotion, inference, scheduling, and monitoring for the blog post: **Split Repo ML on Snowflake: Separating Feature Store and Model Training**

This repo handles the **training side** of the split-repo pattern. It consumes Versioned Datasets published by the [Feature Store repo](https://github.com/jar-ry/snowflake-ds-04-feature-store) and never touches raw tables or feature logic directly.

## Repo Structure

```
ml-training-repo/
├── main.py                             # CLI entrypoint (training | promotion | inference | monitoring | scheduling | all)
├── connection.json.example             # Snowflake connection template
├── conda.yml                           # Conda environment for ML Jobs
├── .gitignore
├── conf/
│   └── parameters.yml                  # All config: Snowflake, modelling, HPO, compute, serving, scheduling, monitoring
├── src/
│   ├── session.py                      # Snowpark session factory
│   ├── modelling/
│   │   ├── train.py                    # ML Job entrypoint for HPO (submitted via submit_directory)
│   │   ├── pipeline.py                 # sklearn Pipeline: ColumnTransformer + XGBRegressor
│   │   ├── splitter.py                 # Load Versioned Dataset, train/val split
│   │   └── evaluate.py                 # MAE, MAPE, R² metrics
│   ├── ml_engineering/
│   │   ├── promotion.py                # Best-version selection, tag + set default
│   │   ├── serving.py                  # SPCS service deployment, batch predictions
│   │   ├── scheduling.py              # Stored procedure + Task for scheduled inference
│   │   └── monitoring.py               # ModelMonitor for drift detection
│   └── utils/
│       ├── helpers.py                  # table_exists utility
│       └── versioning.py              # Auto-increment version helpers
├── pipelines/
│   ├── training_pipeline.py            # submit_directory to compute pool
│   ├── promotion_pipeline.py           # Explain best model + promote
│   ├── inference_pipeline.py           # Deploy service, run predictions, save baseline
│   ├── scheduling_pipeline.py          # Create stored procedure + Task
│   └── monitoring_pipeline.py          # Set up ModelMonitor
└── README.md
```

## Quick Start

```bash
# 1. Copy and fill in your Snowflake credentials
cp connection.json.example connection.json

# 2. Run the full pipeline
python main.py all

# Or run individual stages
python main.py training
python main.py promotion
python main.py inference
python main.py monitoring
python main.py scheduling

# Or run a range
python main.py --from training --to inference
```

## The Contract

The **Versioned Dataset** is the interface between the Feature Store repo and this repo. The Feature Store publishes it; this repo reads it. Neither repo imports code from the other.

The training pipeline reads the latest version of the dataset by name:
```python
ds = Dataset.load(session=session, name="RETAIL_REGRESSION_DEMO.FEATURE_STORE.TRAINING_DATASET")
```

## Setup

See the [setup repo](https://github.com/jar-ry/snowflake-ds-setup) for environment and Snowflake object creation.

## Related Repos

| Repo | Description |
|------|-------------|
| [snowflake-ds-setup](https://github.com/jar-ry/snowflake-ds-setup) | Environment setup, data generation, and helper utilities (run this first) |
| [snowflake-ds-04-feature-store](https://github.com/jar-ry/snowflake-ds-04-feature-store) | Feature Store repo: publishes Versioned Datasets consumed by this repo |
| [snowflake-ds-03-ml-jobs-framework](https://github.com/jar-ry/snowflake-ds-03-ml-jobs-framework) | Single-repo version of this pipeline (Part 4) |
