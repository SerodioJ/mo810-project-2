import argparse
from pathlib import Path
from time import perf_counter

from dasf_seismic.attributes.complex_trace import (
    Envelope,
    InstantaneousFrequency,
    CosineInstantaneousPhase,
)
from dasf.ml.xgboost import XGBRegressor
from dasf.pipeline import Pipeline
from dasf.pipeline.executors import DaskPipelineExecutor

from utils import ZarrDataset, FeaturesJoin, SaveModel, ReshapeLabels, ReshapeFeatures, generate_neighbourhood_features, create_executor

attributes = {
    "ENVELOPE": Envelope,
    "INST-FREQ": InstantaneousFrequency,
    "COS-INST-PHASE": CosineInstantaneousPhase,
}

def create_pipeline(
    executor: DaskPipelineExecutor,
    dataset_path: Path,
    attribute_name: str,
    inline_window: int,
    trace_window: int,
    samples_window: int,
    model_output: int,
    pipeline_save_location: str
) -> Pipeline:
    dataset = ZarrDataset(name="F3 dataset", data_path=dataset_path)
    attribute = attributes[attribute_name]()

    neighbourhood = generate_neighbourhood_features(
        inline_window, trace_window, samples_window
    )

    features_join = FeaturesJoin()
    reshape_features = ReshapeFeatures()
    reshape_labels = ReshapeLabels()
    xgboost = XGBRegressor()
    save_model = SaveModel(model_output)

    pipeline = Pipeline(
        name=f"{attribute_name} XGBoost Training Pipeline", executor=executor
    )
    pipeline.add(dataset)
    pipeline.add(attribute, X=dataset)
    for neighbour in neighbourhood.values():
        pipeline.add(neighbour, X=dataset)
    pipeline.add(features_join, **{"(i,j,k)": dataset, **neighbourhood})
    pipeline.add(reshape_features, X=features_join)
    pipeline.add(reshape_labels, X=reshape_features, y=attribute)
    pipeline.add(xgboost.fit, X=reshape_features, y=reshape_labels)
    pipeline.add(save_model, model=xgboost.fit)

    if pipeline_save_location is not None:
        pipeline.visualize(filename=pipeline_save_location)

    return pipeline

def train_model(args):
    executor = create_executor(args.address)
    print("Creating pipeline...")
    pipeline = create_pipeline(
        executor,
        args.data,
        args.attribute,
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
        "-a",
        "--attribute",
        help="attribute used to train the model",
        type=str,
        default="ENVELOPE",
        choices=["ENVELOPE", "INST-FREQ", "COS-INST-PHASE"],
    )

    parser.add_argument(
        "-d",
        "--data",
        help="path to input seismic data",
        type=Path,
        default="data/F3_train.zarr",
    )

    parser.add_argument(
        "-i",
        "--inline-window",
        help="number of neighbors in inline dimension",
        type=int,
        default=0,
    )

    parser.add_argument(
        "-t",
        "--trace-window",
        help="number of neighbors in trace dimension",
        type=int,
        default=0,
    )

    parser.add_argument(
        "-s",
        "--samples-window",
        help="number of neighbors in samples dimension",
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
        help="name of output file to save trained model",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    train_model(args)
