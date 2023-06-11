import argparse
import csv
import os
from pathlib import Path
from multiprocessing import Process, Value
from time import perf_counter

from train_model import train_model

models_prefix = {
    "ENVELOPE": "Env-ml-model",
    "INST-FREQ": "Inst-Freq-ml-model",
    "COS-INST-PHASE": "CIP-ml-model",
}

configs = [
    {"inline_window": 0, "trace_window": 0, "samples_window": 0},
    {"inline_window": 0, "trace_window": 0, "samples_window": 1},
    {"inline_window": 0, "trace_window": 0, "samples_window": 2},
    {"inline_window": 0, "trace_window": 0, "samples_window": 3},
    {"inline_window": 0, "trace_window": 0, "samples_window": 4},
    {"inline_window": 1, "trace_window": 0, "samples_window": 0},
    {"inline_window": 0, "trace_window": 1, "samples_window": 0},
    {"inline_window": 1, "trace_window": 1, "samples_window": 1},
    {"inline_window": 2, "trace_window": 2, "samples_window": 2},
    {"inline_window": 3, "trace_window": 3, "samples_window": 3},
]


def train(train_args, time):
    time.value = train_model(train_args)


def model_name(prefix, workers, x, y, z):
    return os.path.join("models", workers, f"{prefix}-{x}-{y}-{z}.json")


def evaluate_training(args):
    pipeline_time = Value("d", 0)
    for attr, prefix in models_prefix.items():
        with open(f"train-{attr}-{args.workers}.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "inline_window",
                    "trace_window",
                    "samples_window",
                    "pipeline_time",
                    "process_time",
                ]
            )
            for config in configs:
                train_args = {
                    "attribute": attr,
                    "fig_pipeline": None,
                    "output": model_name(
                        prefix=prefix,
                        workers=args.workers,
                        x=config.inline_window,
                        y=config.trace_window,
                        z=config.samples_window,
                    ),
                    **config,
                    **args,
                }
                pipeline_time.value = 0
                p = Process(
                    target=train,
                    args=(train_args, pipeline_time),
                )
                start = perf_counter()
                p.start()
                p.join()
                time = perf_counter() - start
                writer.writerow(
                    [
                        config.inline_window,
                        config.trace_window,
                        config.samples_window,
                        pipeline_time.value,
                        time,
                    ]
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-w", "--workers", help="number of workers used", type=int, required=True
    )

    parser.add_argument(
        "-d",
        "--data",
        help="path to input seismic data",
        type=Path,
        default="data/F3_train.zarr",
    )

    parser.add_argument(
        "-e",
        "--address",
        help="Dask Scheduler address HOST:PORT",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    run_model(args)
