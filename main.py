import argparse
import sys

import yaml

from src.pipelines import (
    inference_pipeline,
    monitoring_pipeline,
    promotion_pipeline,
    scheduling_pipeline,
    training_pipeline,
)

PIPELINE_ORDER = [
    "training",
    "promotion",
    "inference",
    "monitoring",
    "scheduling",
]

PIPELINES = {
    "training": training_pipeline,
    "promotion": promotion_pipeline,
    "inference": inference_pipeline,
    "monitoring": monitoring_pipeline,
    "scheduling": scheduling_pipeline,
}

PIPELINE_NAMES = ", ".join(PIPELINE_ORDER)


def load_config(config_path: str = "conf/parameters.yml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_range(conf: dict, start: str, end: str):
    start_idx = PIPELINE_ORDER.index(start)
    end_idx = PIPELINE_ORDER.index(end)
    if start_idx > end_idx:
        print(f"Error: --from '{start}' comes after --to '{end}'.")
        print(f"Pipeline order: {PIPELINE_NAMES}")
        sys.exit(1)

    stages = PIPELINE_ORDER[start_idx : end_idx + 1]
    print(f"Running pipelines: {' -> '.join(stages)}\n")

    from src.session import create_session

    session, *_ = create_session(conf)
    for name in stages:
        PIPELINES[name].run(session, conf)

    print("\n" + "=" * 60)
    print(f"PIPELINES COMPLETE: {' -> '.join(stages)}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Customer Value Model Training Pipeline Runner",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "pipeline",
        nargs="?",
        default=None,
        choices=["all", *PIPELINE_ORDER],
        help="Pipeline to run (default: all)\n"
        "  all        - Run full end-to-end pipeline\n"
        "  training   - Submit HPO training ML Job\n"
        "  promotion  - Promote best model version\n"
        "  inference  - Run batch predictions\n"
        "  monitoring - Set up model monitoring\n"
        "  scheduling - Create scheduled inference task (suspended)",
    )
    parser.add_argument(
        "--from",
        dest="from_pipeline",
        choices=PIPELINE_ORDER,
        help=f"Start pipeline (inclusive). Options: {PIPELINE_NAMES}",
    )
    parser.add_argument(
        "--to",
        dest="to_pipeline",
        choices=PIPELINE_ORDER,
        help=f"End pipeline (inclusive). Options: {PIPELINE_NAMES}",
    )
    parser.add_argument(
        "--config",
        "-c",
        default="conf/parameters.yml",
        help="Path to parameters YAML (default: conf/parameters.yml)",
    )
    args = parser.parse_args()

    if args.pipeline and (args.from_pipeline or args.to_pipeline):
        parser.error("Cannot use positional pipeline name with --from/--to.")

    conf = load_config(args.config)
    print(f"Config loaded from: {args.config}")
    print(f"Database: {conf['snowflake']['database']}")

    if args.from_pipeline or args.to_pipeline:
        start = args.from_pipeline or PIPELINE_ORDER[0]
        end = args.to_pipeline or PIPELINE_ORDER[-1]
        print(f"Range: {start} -> {end}\n")
        run_range(conf, start, end)
    elif args.pipeline is None or args.pipeline == "all":
        print("Pipeline: all\n")
        run_range(conf, PIPELINE_ORDER[0], PIPELINE_ORDER[-1])
    else:
        print(f"Pipeline: {args.pipeline}\n")
        from src.session import create_session

        session, *_ = create_session(conf)
        PIPELINES[args.pipeline].run(session, conf)


if __name__ == "__main__":
    main()
