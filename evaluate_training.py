import argparse
import csv
import os
from collections import namedtuple
from pathlib import Path
from multiprocessing import Process, Value
from time import perf_counter, sleep

from train_model import train_model as train_model
from train_model_rebalance import train_model as t_rebalance

models_prefix = {
    # "ENVELOPE": "Env-ml-model",
    # "INST-FREQ": "Inst-Freq-ml-model",
    "COS-INST-PHASE": "CIP-ml-model",
}

configs = [
    {"inline_window": 0, "trace_window": 0, "samples_window": 0},
    {"inline_window": 0, "trace_window": 0, "samples_window": 4},
    {"inline_window": 2, "trace_window": 2, "samples_window": 2},
    {"inline_window": 4, "trace_window": 4, "samples_window": 4},
]

modules = {
    "baseline": train_model,
    "rebalance": t_rebalance,
}


def train(train_args, time, func):
    time.value = func(train_args)


def model_name(module, prefix, workers, x, y, z, ext="json"):
    return os.path.join("models", str(workers), f"{module}-{prefix}-{x}-{y}-{z}.{ext}")


def evaluate_training(args):
    pipeline_time = Value("d", 0)
    for module, func in modules.items():
        for attr, prefix in models_prefix.items():
            with open(f"train-{module}-{attr}-{args.workers}.csv", "w") as f:
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
                    print(config)
                    for sample in range(3):
                        print(f"{sample} SAMPLE")
                        train_args = {
                            "attribute": attr,
                            "fig_pipeline": None,
                            "output": model_name(
                                module=f"{module}_{sample}",
                                prefix=prefix,
                                workers=args.workers,
                                x=config["samples_window"],
                                y=config["trace_window"],
                                z=config["inline_window"],
                            ),
                            "data": args.data,
                            "address": args.address,
                            "fig_pipeline": model_name(
                                module=module,
                                prefix=prefix,
                                workers=args.workers,
                                x=config["samples_window"],
                                y=config["trace_window"],
                                z=config["inline_window"],
                                ext="png",
                            ),
                            "report": f"{module}-w{args.workers}.{sample}",
                            "workers": args.workers,
                            **config,
                        }
                        train_args = namedtuple("args", train_args.keys())(
                            *train_args.values()
                        )
                        pipeline_time.value = 0
                        p = Process(
                            target=train,
                            args=(train_args, pipeline_time, func),
                        )
                        sleep(10)
                        start = perf_counter()
                        p.start()
                        p.join()
                        time = perf_counter() - start
                        writer.writerow(
                            [
                                config["inline_window"],
                                config["trace_window"],
                                config["samples_window"],
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
        required=False,
        default=None,
    )

    args = parser.parse_args()

    evaluate_training(args)
