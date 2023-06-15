import argparse
from pathlib import Path
from time import perf_counter

from dasf_seismic.attributes.complex_trace import (
    Envelope,
    InstantaneousFrequency,
    CosineInstantaneousPhase,
)
from dasf.pipeline import Pipeline
from dasf.pipeline.executors import DaskPipelineExecutor

from utils import ZarrDataset, CreateDataFrame, generate_neighbourhood_features, create_executor

def create_pipeline(
    executor: DaskPipelineExecutor,
    dataset_path: Path,
    inline_window: int,
    trace_window: int,
    samples_window: int,
    output: int,
    pipeline_save_location: str
) -> Pipeline:
    dataset = ZarrDataset(name="F3 dataset", data_path=dataset_path)
    
    attributes = {
        "env": Envelope(),
        "freq": InstantaneousFrequency(),
        "cos": CosineInstantaneousPhase(),
    }

    neighbourhood = generate_neighbourhood_features(
        inline_window, trace_window, samples_window
    )

    features_join = CreateDataFrame(fname=output)

    pipeline = Pipeline(
        name="Feature Extraction Pipeline", executor=executor
    )
    pipeline.add(dataset)
    for neighbour in neighbourhood.values():
        pipeline.add(neighbour, X=dataset)
    for attribute in attributes.values():
        pipeline.add(attribute, X=dataset)
    pipeline.add(features_join, **{"(i,j,k)": dataset, **neighbourhood, **attributes})

    if pipeline_save_location is not None:
        pipeline._dag_g.render(outfile=pipeline_save_location, cleanup=True)

    return pipeline

def create_dataframe(args):
    executor = create_executor(args.address)
    executor.client.upload_file("utils.py")
    
    print("Creating pipeline...")
    pipeline = create_pipeline(
        executor,
        args.data,
        args.inline_window,
        args.trace_window,
        args.samples_window,
        args.output,
        args.fig_pipeline
    )

    print("Executing pipeline...")
    start = perf_counter()
    pipeline.run()
    end = perf_counter()
    
    print(f"Done! Execution time: {end - start:.2f} s")
    return end - start


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

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
        default=None
    )

    parser.add_argument(
        "-f",
        "--fig-pipeline",
        help="name of file to save pipeline figure",
        type=str,
        default=None
    )

    parser.add_argument(
        "-o",
        "--output",
        help="name of output file to save dataframe",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    create_dataframe(args)
