import argparse
from pathlib import Path
from time import perf_counter

from dasf_seismic.attributes.complex_trace import (
    Envelope,
    InstantaneousFrequency,
    CosineInstantaneousPhase,
)
from dasf.transforms import ArraysToDataFrame
from dasf.ml.xgboost import XGBRegressor
from dasf.pipeline import Pipeline
from dasf.pipeline.executors import DaskPipelineExecutor

from utils import (
    ZarrDataset,
    SaveModel,
    Repartition,
    SplitFeatures,
    SplitLabel,
    RebalanceData,
    generate_neighbourhood_features,
    create_executor,
)

attributes = {
    Envelope: {
        "model_kwargs": {
            "learning_rate": 0.5,
            "n_estimator": 10,
            "max_depth": 5,
            "booster": "gbtree",
        },
        "window": (5, 0, 0),
        "name": "Env-ml-model-5-0-0.json",
    },
    InstantaneousFrequency: {
        "model_kwargs": {
            "learning_rate": 0.5,
            "n_estimator": 10,
            "max_depth": 5,
            "booster": "gbtree",
        },
        "window": (5, 1, 1),
        "name": "Inst-Freq-ml-model-5-1-1.json",
    },
    CosineInstantaneousPhase: {
        "model_kwargs": {
            "learning_rate": 0.5,
            "n_estimator": 10,
            "max_depth": 10,
            "booster": "gbtree",
        },
        "window": (5, 0, 0),
        "name": "CIP-ml-model-5-0-0.json",
    },
}


def create_pipeline(
    executor: DaskPipelineExecutor,
    dataset_path: Path,
    attribute,
    inline_window: int,
    trace_window: int,
    samples_window: int,
    model_output: str,
    model_kwargs: dict,
) -> Pipeline:
    dataset = ZarrDataset(name="F3 dataset", data_path=dataset_path)
    attribute = attribute()

    neighbourhood = generate_neighbourhood_features(
        inline_window, trace_window, samples_window
    )
    features = {"(i,j,k)": dataset, **neighbourhood, "label": attribute}

    features_join = ArraysToDataFrame()
    repartition = Repartition(96)
    rebalance = RebalanceData(
        executor.client, f"{model_output.split('.')[0]}-chunks.json"
    )
    label = SplitLabel()
    feat = SplitFeatures(feature_set=list(features.keys())[:-1])
    xgboost = XGBRegressor(**model_kwargs)
    save_model = SaveModel(model_output)

    pipeline = Pipeline(name=f"Best Model XGBoost Training Pipeline", executor=executor)
    pipeline.add(dataset)
    pipeline.add(attribute, X=dataset)
    for neighbour in neighbourhood.values():
        pipeline.add(neighbour, X=dataset)
    pipeline.add(features_join, **features)
    pipeline.add(features_join, **features)
    pipeline.add(repartition, X=features_join)
    pipeline.add(rebalance, X=repartition)
    pipeline.add(label, X=rebalance)
    pipeline.add(feat, X=rebalance)
    pipeline.add(xgboost.fit, X=feat, y=label)
    pipeline.add(save_model, model=xgboost.fit)

    return pipeline


def train_models(args):
    for attr, config in attributes.items():
        executor = create_executor(args.address)
        executor.client.upload_file("utils.py")
        print("Creating pipeline...")
        pipeline = create_pipeline(
            executor,
            args.data,
            attr,
            config["window"][2],
            config["window"][1],
            config["window"][0],
            config["name"],
            config["model_kwargs"],
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
        "-e",
        "--address",
        help="Dask Scheduler address HOST:PORT",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    train_models(args)
