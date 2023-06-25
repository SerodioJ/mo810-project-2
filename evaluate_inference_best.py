import argparse
import csv
import os
from collections import namedtuple
from glob import glob
from pathlib import Path
from multiprocessing import Process, Value
from time import perf_counter

import dask.array as da
import zarr
from sklearn.metrics import r2_score
from dasf_seismic.attributes.complex_trace import (
    Envelope,
    InstantaneousFrequency,
    CosineInstantaneousPhase,
)

from run_model import run_model


attributes = {
    "ENVELOPE": (Envelope, "Env-ml-model-5-0-0.json"),
    "INST-FREQ": (InstantaneousFrequency, "Inst-Freq-ml-model-5-1-1.json"),
    "COS-INST-PHASE": (CosineInstantaneousPhase, "CIP-ml-model-5-0-0.json"),
}


def run(run_args, attribute, time, save, r2):
    predict, total = run_model(run_args)
    time.value = predict
    save.value = total
    data = da.from_zarr(run_args.data, chunks={0: "auto", 0: "auto", 0: -1})
    attr = attributes[attribute][0]()
    y_hat = attr.transform(data)
    y_pred = zarr.load(run_args.output)
    r2.value = r2_score(y_hat.flatten(), y_pred.flatten())


def get_windows(model_file):
    file_name = model_file.split("/")[-1]
    file_name = file_name.split(".")[0]
    contents = file_name.split("-")
    return {
        "inline_window": int(contents[-1]),
        "trace_window": int(contents[-2]),
        "samples_window": int(contents[-3]),
    }


def model_name(prefix, workers, x, y, z, ext="png"):
    return os.path.join("models", str(workers), f"{prefix}-run-{x}-{y}-{z}.{ext}")


def evaluate_inference(args):
    predict_time = Value("d", 0)
    save_time = Value("d", 0)
    r2 = Value("d", 0)
    with open(f"run-best.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "id",
                "inline_window",
                "trace_window",
                "samples_window",
                "predict_time",
                "save_time",
                "process_time",
                "r2_score",
            ]
        )
        for attr, info in attributes.items():
            model = info[1]
            print(model)
            windows = get_windows(model)
            run_args = {
                "ml_model": model,
                "fig_pipeline": None,
                "output": os.path.join("data", "pred2.zarr"),
                "data": args.data,
                "address": args.address,
                "fig_pipeline": model_name(
                    prefix=f"best-{attr}",
                    workers=args.workers,
                    x=windows["samples_window"],
                    y=windows["trace_window"],
                    z=windows["inline_window"],
                ),
                "report": f"w{args.workers}-{attr}",
                **windows,
            }
            run_args = namedtuple("args", run_args.keys())(*run_args.values())
            predict_time.value = 0
            save_time.value = 0
            r2.value = 0
            p = Process(
                target=run,
                args=(run_args, attr, predict_time, save_time, r2),
            )
            start = perf_counter()
            p.start()
            p.join()
            time = perf_counter() - start
            writer.writerow(
                [
                    model,
                    windows["inline_window"],
                    windows["trace_window"],
                    windows["samples_window"],
                    predict_time.value,
                    save_time.value,
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

    evaluate_inference(args)
