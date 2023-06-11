import argparse
import csv
import os
from glob import glob
from pathlib import Path
from multiprocessing import Process, Value
from time import perf_counter

import zarr
from sklearn.metrics import r2_score
from dasf_seismic.attributes.complex_trace import (
    Envelope,
    InstantaneousFrequency,
    CosineInstantaneousPhase,
)

from run_model import run_model

models_prefix = {
    "ENVELOPE": "Env-ml-model",
    "INST-FREQ": "Inst-Freq-ml-model",
    "COS-INST-PHASE": "CIP-ml-model",
}

attributes = {
    "ENVELOPE": Envelope,
    "INST-FREQ": InstantaneousFrequency,
    "COS-INST-PHASE": CosineInstantaneousPhase,
}


def run(run_args, attribute, time, r2):
    time.value = run_model(run_args)
    data = zarr.load(run_args.data)
    attr = attributes[attribute]()
    y_hat = attr.transform(data)
    y_pred = zarr.load(run_args.output)
    r2.value = r2_score(y_hat, y_pred)


def get_models(workers, prefix):
    return glob(os.path.join("models", workers, f"{prefix}-*.json"))


def get_windows(model_file):
    file_name = model_file.split("/")[-1]
    file_name = file_name.split(".")[0]
    contents = file_name.split("-")
    return {
        "inline_window": int(contents[1]),
        "trace_window": int(contents[2]),
        "samples_window": int(contents[3]),
    }


def evaluate_training(args):
    pipeline_time = Value("d", 0)
    r2 = Value("d", 0)
    for attr, prefix in models_prefix.items():
        with open(f"run-{attr}-{args.workers}.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "inline_window",
                    "trace_window",
                    "samples_window",
                    "pipeline_time",
                    "process_time",
                    "r2_score",
                ]
            )
            for model in get_models(args.workers, prefix):
                windows = get_windows(model)
                run_args = {
                    "ml_model": model,
                    "fig_pipeline": None,
                    "output": os.path.join("data", "pred.zarr"),
                    **windows,
                    **args,
                }
                pipeline_time.value = 0
                r2.value = 0
                p = Process(
                    target=run,
                    args=(run_args, attr, pipeline_time, r2),
                )
                start = perf_counter()
                p.start()
                p.join()
                time = perf_counter() - start
                writer.writerow(
                    [
                        windows.inline_window,
                        windows.trace_window,
                        windows.samples_window,
                        pipeline_time.value,
                        time,
                        r2.value,
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
