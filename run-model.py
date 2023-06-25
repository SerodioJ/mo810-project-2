import argparse
from pathlib import Path
from time import perf_counter

from dask.distributed import performance_report
from dasf.transforms import ArraysToDataFrame
from dasf.ml.xgboost import XGBRegressor
from dasf.pipeline import Pipeline
from dasf.pipeline.executors import DaskPipelineExecutor
from dasf.utils.funcs import is_gpu_supported

from utils import (
    ZarrDataset,
    SaveResult,
    # Repartition,
    generate_neighbourhood_features,
    create_executor,
)


def load_model(fname):
    xgboost = XGBRegressor()
    xgboost._XGBRegressor__xgb_cpu.load_model(fname)
    xgboost._XGBRegressor__xgb_mcpu.load_model(fname)
    if is_gpu_supported():
        xgboost._XGBRegressor__xgb_gpu.load_model(fname)
        xgboost._XGBRegressor__xgb_mgpu.load_model(fname)
    return xgboost


def create_pipeline(
    executor: DaskPipelineExecutor,
    dataset_path: Path,
    ml_model: Path,
    inline_window: int,
    trace_window: int,
    samples_window: int,
    output: str,
    pipeline_save_location: str,
) -> Pipeline:

    dataset = ZarrDataset(name="F3 dataset", data_path=dataset_path)

    neighbourhood = generate_neighbourhood_features(
        inline_window, trace_window, samples_window
    )

    features_join = ArraysToDataFrame()
    xgboost = load_model(ml_model)
    # repartition = Repartition(96)
    save_result = SaveResult(output)

    pipeline = Pipeline(
        name=f"XGBoost({ml_model}) Inference Pipeline", executor=executor
    )
    pipeline.add(dataset)
    for neighbour in neighbourhood.values():
        pipeline.add(neighbour, X=dataset)
    pipeline.add(features_join, **{"(i,j,k)": dataset, **neighbourhood})
    # pipeline.add(repartition, X=features_join)
    pipeline.add(xgboost.predict, X=features_join)
    pipeline.add(save_result, X=xgboost.predict, raw=dataset)

    if pipeline_save_location is not None:
        pipeline._dag_g.render(outfile=pipeline_save_location, cleanup=True)

    return pipeline, save_result


def run_model(args):
    executor = create_executor(args.address)
    executor.client.upload_file("utils.py")

    print("Creating pipeline...")
    pipeline, save = create_pipeline(
        executor,
        args.data,
        args.ml_model,
        args.inline_window,
        args.trace_window,
        args.samples_window,
        args.output,
        args.fig_pipeline,
    )

    print("Executing pipeline...")
    if args.report:
        with performance_report(
            filename=f"run-{args.report}-{args.samples_window}-{args.trace_window}-{args.inline_window}.html"
        ):
            start = perf_counter()
            pipeline.run()
            end = perf_counter()
    else:
        start = perf_counter()
        pipeline.run()
        end = perf_counter()
    compute_time = pipeline.get_result_from(save)
    print(f"Done! Execution time: {end - start:.2f} s")
    return compute_time, end - start


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m", "--ml-model", help="trained model file", type=Path, required=True
    )

    parser.add_argument(
        "-d",
        "--data",
        help="path to input seismic data",
        type=Path,
        default="data/F3_train.zarr",
    )

    parser.add_argument(
        "-x",
        "--samples-window",
        help="number of neighbors in samples dimension",
        type=int,
        default=0,
    )

    parser.add_argument(
        "-y",
        "--trace-window",
        help="number of neighbors in trace dimension",
        type=int,
        default=0,
    )

    parser.add_argument(
        "-z",
        "--inline-window",
        help="number of neighbors in inline dimension",
        type=int,
        default=0,
    )

    parser.add_argument(
        "-e",
        "--address",
        help="Dask Scheduler address HOST:PORT",
        type=str,
        default=None,
    )

    parser.add_argument(
        "-o",
        "--output",
        help="name of output file to save generated seismic aatribute",
        type=str,
        required=True,
    )

    parser.add_argument(
        "-r",
        "--report",
        help="report base name to generate file",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    run_model(args)
