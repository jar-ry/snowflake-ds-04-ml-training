# ML Training Repo — Agent Guide

## What This Repo Is

The **model training side** of a split-repo ML pattern. This repo consumes Versioned Datasets published by the Feature Store repo and handles training, promotion, inference, scheduling, and monitoring. It never touches raw tables or feature engineering logic.

The contract between this repo and the Feature Store repo is a **Versioned Dataset**.

**Use case:** Customer value regression (predict `MONTHLY_CUSTOMER_VALUE`).

## Repo Structure

```
├── main.py                          # CLI entrypoint (training | promotion | inference | monitoring | scheduling | all)
├── conda.yml                        # Conda environment
├── conf/
│   └── parameters.yml               # All pipeline configuration
├── pipelines/
│   ├── training_pipeline.py         # Submit HPO training job via submit_directory
│   ├── promotion_pipeline.py        # Explain best model + promote (alias, tags, default)
│   ├── inference_pipeline.py        # Batch inference via model version
│   ├── scheduling_pipeline.py       # Scheduled batch inference via stored procedure
│   └── monitoring_pipeline.py       # ModelMonitor for drift detection
├── src/
│   ├── session.py                   # Snowpark session factory
│   ├── modelling/
│   │   ├── train.py                 # ML Job entrypoint for HPO (submit_directory target)
│   │   ├── pipeline.py              # sklearn Pipeline (ColumnTransformer + XGBRegressor)
│   │   ├── splitter.py              # Load Versioned Dataset, train/val split
│   │   └── evaluate.py             # MAE, MAPE, R² metrics
│   ├── ml_engineering/
│   │   ├── promotion.py             # Best-version selection, tag + set default
│   │   ├── serving.py               # SPCS service deployment, batch predictions
│   │   ├── scheduling.py            # Stored procedure + Task for scheduled inference
│   │   └── monitoring.py            # ModelMonitor setup
│   └── utils/
│       ├── helpers.py               # table_exists utility
│       └── versioning.py            # Auto-increment version helpers
└── connection.json.example          # Snowflake credentials template
```

## Environment

```bash
conda env create -f conda.yml
conda activate snowflake_ds
```

Python 3.10. Key packages: `snowflake-ml-python>=1.30.0`, `xgboost`, `scikit-learn`, `altair`.

## How to Run

```bash
python main.py all                        # Full end-to-end
python main.py training                   # Submit HPO job
python main.py promotion                  # Explain + promote best model
python main.py inference                  # Deploy + batch predict
python main.py scheduling                 # Create scheduled task
python main.py monitoring                 # Set up drift monitoring
python main.py --from training --to inference   # Run a range
```

## Snowflake Connection

`src/session.py` reads `connection.json` (copy from `connection.json.example`).

Environment variable override: set `SNOWFLAKE_CONNECTION_NAME` to use a named connection.

Inside ML Job containers, `Session.builder.getOrCreate()` provides the session automatically.

## Configuration

All parameters live in `conf/parameters.yml`. Same structure as the single-repo framework (Part 4). Key sections:

- **snowflake** — database, schema, role, warehouse
- **feature_store** — schema, FeatureView name, dataset name (read-only — this repo consumes, never writes)
- **model_registry** — schema for versioned models
- **modelling** — model name, feature/target columns, column types, encoders, tuning metric
- **hpo** — hyperparameter search space
- **compute** — pool name, stage, target instances, num trials
- **serving** — inference service config
- **scheduling** — stored procedure + Task definition
- **monitoring** — prediction/baseline tables, refresh intervals

## Key Snowflake Objects

- **Database:** `RETAIL_REGRESSION_DEMO`
- **Schemas:** `DS`, `MODELLING`, `FEATURE_STORE`
- **Compute Pool:** `CUSTOMER_VALUE_MODEL_POOL_CPU`
- **Model:** `UC01_SNOWFLAKEML_RF_REGRESSOR_MODEL`
- **Dataset:** `TRAINING_DATASET` (consumed, not created — published by Feature Store repo)
- **Stage:** `payload_stage`

## Architecture Notes

- `submit_directory` ships the entire project to the compute pool. `src/modelling/train.py` is the entrypoint.
- `train.py` has two roles: (1) the `train()` function is the per-trial HPO function run by Ray workers, (2) the `__main__` block sets up the Tuner and launches HPO.
- `SnowflakeXgboostCallback` is commented out — it doesn't support `target_platforms` or `enable_explainability`. Models are logged via `exp.log_model()` with `target_platforms=["WAREHOUSE", "SNOWPARK_CONTAINER_SERVICES"]` and `options={"enable_explainability": True}`.
- Before HPO, the `__main__` block pre-creates the model in the Registry with a dummy version to avoid "Object already exists" race conditions from parallel trials.
- `promotion_pipeline.py` runs explainability (SHAP) on the best model before promoting it.
- This repo reads from a Versioned Dataset by name — it never imports feature logic or touches raw tables.

## The Contract

The Feature Store repo publishes a **Versioned Dataset**. This repo reads it by name and version. Neither repo imports code from the other.

```python
ds = Dataset.load(session=session, name="RETAIL_REGRESSION_DEMO.FEATURE_STORE.TRAINING_DATASET")
```

## Common Modifications

- **Change model type:** Edit `src/modelling/pipeline.py`, update `hpo` section in `parameters.yml`, update `src/modelling/evaluate.py`
- **Change HPO:** Modify `hpo` section in `parameters.yml` (parameter names must match model constructor args)
- **Change compute:** Adjust `compute` section in `parameters.yml`
- **Add a pipeline stage:** Create `pipelines/new_pipeline.py` with a `run(session, conf)` function, register in `main.py`
- **Refactor for a different use case:** Use the `refactor-framework` Cortex Code skill (`.cortex/skills/refactor-framework/SKILL.md`)
