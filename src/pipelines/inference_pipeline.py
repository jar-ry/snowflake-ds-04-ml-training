from src.ml_engineering.serving import deploy_inference_service, run_batch_predictions
from src.utils.helpers import get_or_create_registry, table_exists


def run(session, conf: dict):
    print("=" * 60)
    print("INFERENCE PIPELINE")
    print("=" * 60)

    database = conf["snowflake"]["database"]
    mr_schema = conf["model_registry"]["schema"]
    model_name = conf["modelling"]["model_name"]
    prediction_table = conf["monitoring"]["prediction_table"]
    baseline_table = conf["monitoring"].get("baseline_table")
    pool_name = conf["compute"]["pool_name"]
    service_name = conf["serving"]["service_name"]

    fs_schema = conf["feature_store"]["schema"]
    feature_views = conf["feature_store"].get("feature_views", [])
    if feature_views:
        fv_tables = [
            f"{database}.{fs_schema}.{fv['name']}${fv['version']}" for fv in feature_views
        ]
        input_df = session.table(fv_tables[0])
        for fv_table in fv_tables[1:]:
            right_df = session.table(fv_table)
            join_cols = list(set(input_df.columns) & set(right_df.columns))
            right_cols = [c for c in right_df.columns if c not in join_cols]
            input_df = input_df.join(right_df.select(join_cols + right_cols), on=join_cols)
        input_source = ", ".join(fv_tables)
    else:
        fv_name = conf["feature_store"]["feature_view_name"]
        fv_version = conf["feature_store"]["feature_view_version"]
        input_source = f"{database}.{fs_schema}.{fv_name}${fv_version}"
        input_df = session.table(input_source)

    mr = get_or_create_registry(session, database, mr_schema)

    model = mr.get_model(model_name)
    default_version = model.default
    if default_version is None:
        print("No default model version set. Run promotion pipeline first.")
        return None

    version_name = default_version.version_name
    print(f"  Model   : {model_name}")
    print(f"  Version : {version_name} (default)")
    print(f"  Input   : {input_source}")
    print(f"  Output  : {prediction_table}")
    print(f"  Service : {service_name}")
    print(f"  Pool    : {pool_name}")

    print("\n[1/3] Deploying inference service...")
    deploy_inference_service(session, mr, model_name, version_name, pool_name, service_name)

    print("[2/3] Running batch predictions via SPCS...")
    run_batch_predictions(session, mr, model_name, input_df, prediction_table, service_name)

    row_count = session.table(prediction_table).count()
    print(f"  {row_count} predictions written.")

    print("[3/3] Checking baseline table...")
    if baseline_table:
        if not table_exists(session, baseline_table):
            session.table(prediction_table).write.mode("overwrite").save_as_table(baseline_table)
            print(f"  Baseline snapshot saved to {baseline_table} ({row_count} rows)")
        else:
            print(f"  Baseline already exists at {baseline_table}, skipping.")
    else:
        print("  No baseline_table configured, skipping.")

    print("\nInference pipeline complete.")
