---
name: refactor-framework
description: "Refactor this ML Training repo for a new use case. Use when: adapting the pipeline for a different model, dataset, or business problem. Triggers: refactor, adapt, new use case, change model, swap dataset, customise, customize, churn, fraud, forecasting, classification, new project, my own data."
---

# Refactor ML Training

You are helping a user adapt this Snowflake ML Training repo to their own use case. This repo consumes Versioned Datasets published by the Feature Store repo and handles everything from HPO training through deployment and monitoring via `submit_directory`.

## Repo Architecture

```
ml-training-repo/
├── main.py                             # CLI entrypoint (training | promotion | inference | monitoring | scheduling | all)
├── conda.yml                           # Conda environment for ML Jobs
├── conf/
│   └── parameters.yml                  # All config: Snowflake, modelling, HPO, compute, serving, scheduling, monitoring
└── src/
    ├── session.py                      # Snowpark session factory
    ├── pipelines/
    │   ├── training_pipeline.py        # submit_directory to compute pool
    │   ├── promotion_pipeline.py       # Find best model, promote
    │   ├── inference_pipeline.py       # Deploy service, run predictions, save baseline
    │   ├── scheduling_pipeline.py      # Create stored procedure + Task
    │   └── monitoring_pipeline.py      # Set up ModelMonitor
    ├── modelling/
    │   ├── train.py                    # ML Job entrypoint for HPO (submitted via submit_directory)
    │   ├── pipeline.py                 # sklearn Pipeline: ColumnTransformer + XGBRegressor
    │   ├── splitter.py                 # Load Versioned Dataset, train/val split
    │   └── evaluate.py                 # MAE, MAPE, R² metrics
    ├── ml_engineering/
    │   ├── promotion.py                # Best-version selection, tag + set default
    │   ├── serving.py                  # SPCS service deployment, batch predictions
    │   ├── scheduling.py              # Stored procedure + Task for scheduled inference
    │   └── monitoring.py               # ModelMonitor for drift detection
    └── utils/
        ├── helpers.py                  # table_exists utility
        └── versioning.py              # Auto-increment version helpers
```

## Step 1: Gather Requirements

**Goal:** Understand the user's new use case before touching any code.

**Ask these questions (use ask_user_question tool):**

1. **Use case**: What are you building? (e.g. churn prediction, fraud detection, demand forecasting, recommendation scoring)
2. **Model type**: Classification or regression? What algorithm? (e.g. XGBClassifier, LightGBM, RandomForest — the current framework uses XGBRegressor)
3. **Dataset name**: What is the Versioned Dataset name published by the Feature Store repo? (e.g. `RETAIL_REGRESSION_DEMO.FEATURE_STORE.TRAINING_DATASET`)
4. **Target column**: What column are you predicting?
5. **Feature columns**: Which columns are features? Which are numerical, categorical, ordinal? (If the user is unsure, offer to inspect the dataset.)
6. **Snowflake objects**: What database, schema, warehouse, compute pool, and role should the pipeline use? (Offer to keep the existing ones if experimenting.)

**STOP**: Confirm the requirements with the user before proceeding. Summarise what you understood and ask for corrections.

## Step 2: Refactor Configuration

**Goal:** Rewrite `conf/parameters.yml` for the new use case.

**File:** `conf/parameters.yml`

**Changes:**
- `snowflake.*` — Update database, schema, role, warehouse to user's values
- `feature_store.*` — Update dataset name, feature view references
- `model_registry.schema` — Update if needed
- `modelling.*` — New model name, experiment name, target column, feature/numerical/categorical/ordinal columns, ordinal categories, tuning metric (change to appropriate metric for classification vs regression)
- `hpo.*` — Update hyperparameter search space to match the new algorithm (e.g. XGBClassifier params differ from XGBRegressor)
- `compute.*` — Keep or adjust pool name, stage, instances, trials
- `serving.*` — Update service name
- `scheduling.*` — Update task/procedure names
- `monitoring.*` — Update prediction/actual columns, segment columns, table names

**Important:** The `tuning_metric` and `tuning_mode` must match the problem type:
- Regression: `mean_absolute_percentage_error` / `min`, or `r2_score` / `max`
- Classification: `f1_score` / `max`, `accuracy` / `max`, `roc_auc` / `max`, `log_loss` / `min`

## Step 3: Refactor Modelling

**Goal:** Update the model pipeline for the new algorithm and problem type.

### `src/modelling/pipeline.py`
- Replace `xgb.XGBRegressor` with the user's chosen model (e.g. `xgb.XGBClassifier`, `lightgbm.LGBMClassifier`)
- Update the ColumnTransformer if column types changed
- If the user wants a completely different preprocessing approach (e.g. StandardScaler instead of MinMaxScaler), update here
- Update imports

### `src/modelling/evaluate.py`
- Replace regression metrics (MAE, MAPE, R²) with appropriate metrics:
  - Classification: accuracy, precision, recall, f1, roc_auc, log_loss
  - Regression: MAE, MAPE, RMSE, R²
- Update the return dict keys — these must match what `train.py` reports and what `promotion.py` uses to find the best version

### `src/modelling/train.py`
- Update imports if the model library changed (e.g. `import lightgbm` instead of `import xgboost`)
- The HPO search space is built dynamically from `parameters.yml`, so it should mostly work — but verify the parameter names match the new model's API
- If the model requires different fitting logic (e.g. `eval_set` for early stopping), update the `train()` function

### `src/modelling/splitter.py`
- Usually no changes needed — it's generic
- If the user needs stratified splitting (common for imbalanced classification), add `stratify=y` to `train_test_split`

## Step 4: Refactor Downstream Pipelines

**Goal:** Update inference, scheduling, and monitoring for the new use case.

### `src/ml_engineering/promotion.py`
- Update the default `metric` parameter in `get_best_model_version()` to match the new tuning metric from config

### `src/ml_engineering/serving.py`
- Usually no changes — it's generic (calls `mv.run()`)
- If the user needs different prediction column naming, update `prediction_column` parameter

### `src/ml_engineering/monitoring.py`
- Config-driven — should work if `parameters.yml` is correct
- Verify column names match the new prediction output

### `src/ml_engineering/scheduling.py`
- Config-driven — should work if `parameters.yml` is correct

## Step 5: Validate the Versioned Dataset Contract

**Critical:** This repo reads from a Versioned Dataset published by the Feature Store repo. Verify:

- [ ] The dataset name in `conf/parameters.yml` matches what the Feature Store repo publishes
- [ ] All columns listed in `modelling.feature_columns` exist in the dataset
- [ ] The `modelling.target_column` exists in the dataset
- [ ] `modelling.numerical_features` and `modelling.categorical_features` cover all feature columns
- [ ] Column names are consistent between repos (case-sensitive in Snowflake)

**If the Feature Store repo exists alongside this repo**, cross-reference:
1. Read the feature repo's `source.yaml` for the relevant domain
2. Check feature view definitions in `features/*.py` for output column names
3. Verify the entity join key matches

## Step 6: Update Dependencies

**Goal:** Ensure `conda.yml` includes any new libraries.

**Changes:**
- If switching from XGBoost to LightGBM, update the dependency
- If adding new sklearn components, verify they're in the base `scikit-learn` package
- For classification metrics that need `sklearn.metrics`, no new deps needed

## Step 7: Validate

**Goal:** Verify the refactored repo is consistent.

**Checklist:**
- [ ] All column names in `parameters.yml` match the Versioned Dataset schema
- [ ] `pipeline.py` model class matches the algorithm in `parameters.yml`
- [ ] `evaluate.py` metrics match `tuning_metric` in config
- [ ] `train.py` HPO param names match the model's constructor args
- [ ] `conda.yml` includes all required packages
- [ ] `promotion.py` metric name matches `evaluate.py` output

**Actions:**
1. Read through each modified file and cross-reference column names against `parameters.yml`
2. If the user has dataset access, run `SELECT * FROM dataset LIMIT 5` to verify schemas
3. Present a summary of all changes to the user

## Step 8: Test Run

**Suggest:**
```bash
# Test training first (validates dataset loading + model + HPO)
python main.py training

# Then promotion (validates metric lookup)
python main.py promotion

# Then the rest
python main.py --from inference --to monitoring
```

If errors occur, help debug by reading logs and tracing the issue back to the relevant file.

## Important Notes

- **Never hardcode values** — everything goes in `conf/parameters.yml`
- **Keep the pipeline orchestration pattern** — individual pipelines in `src/pipelines/`, business logic in `src/` domain packages
- **Preserve the `submit_directory` pattern** — `train.py` runs inside a Snowflake container, so it must load config from `conf/parameters.yml` at runtime
- **The `train()` function in `train.py` is called by Ray workers** — imports must happen inside the function, not at module level (path issues in distributed execution)
- **This repo never touches raw tables** — it reads from Versioned Datasets only. If the user needs new features, direct them to the Feature Store repo.
- **The Versioned Dataset is the contract** — column name changes must be coordinated with the Feature Store repo
