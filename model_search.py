import argparse
import itertools
import json
import csv
from pathlib import Path
from time import perf_counter
from multiprocessing import Process, Value
from tqdm import tqdm

import numpy as np


from dasf.transforms import PersistDaskData
from dasf.ml.xgboost import XGBRegressor
from dasf.pipeline import Pipeline
from dasf.pipeline.executors import DaskPipelineExecutor

from utils import (
    LoadDataFrame,
    SampleDataframe,
    TrainDataset,
    TestDataset,
    SplitFeatures,
    SplitLabel,
    EvaluateModel,
    DumpToCSV,
    create_executor,
)


def get_features(x, y, z):
    def feature_name(dim, pos):
        name = ["i", "j", "k"]
        name[dim] = f"{name[dim]}{'+' if pos > 0 else '-'}{np.abs(pos)}"
        return f"({','.join(name)})"

    columns = ["(i,j,k)"]
    for dim, n in zip([2, 1, 0], [x, y, z]):
        for pos in range(-n, n + 1):
            if pos == 0:
                continue
            columns.append(feature_name(dim, pos))
    return columns


def create_pipeline(
    executor: DaskPipelineExecutor,
    attr,
    dataframe_path: Path,
    feature_sets: list,
    model_name: str,
    model_kwargs: dict,
    pipeline_save_location: str,
    csv_file: str,
) -> Pipeline:
    models = {}
    features = {}
    labels = {}
    metrics = {}
    for feature_set in feature_sets:
        features_id = f"{feature_set['x']}-{feature_set['y']}-{feature_set['z']}"
        curr_feature_set = get_features(
            feature_set["x"], feature_set["y"], feature_set["z"]
        )
        features[features_id] = [
            SplitFeatures(feature_set=curr_feature_set) for _ in range(4)
        ]
        models[f"{attr}-{features_id}"] = [
            XGBRegressor(**model_kwargs) for _ in range(4)
        ]
        labels[f"{attr}-{features_id}"] = [SplitLabel(label=attr) for _ in range(4)]
        metrics[f"{attr}-{features_id}"] = [
            EvaluateModel(feature_set=curr_feature_set, label=attr)
            for _ in range(4)
        ]

    dataframe = LoadDataFrame(fname=dataframe_path)
    samples = SampleDataframe(split=[0.005, 0.005, 0.005, 0.005, 0.01, 0.97])

    train_datasets = [TrainDataset(index) for index in range(4)]
    persist_train = [PersistDaskData() for _ in range(4)]
    test_dataset = TestDataset(4)
    persist_test = PersistDaskData()

    dump = DumpToCSV(fname=csv_file, model_name=model_name, keys=metrics.keys(), num=4)

    pipeline = Pipeline(name="Hiperparameter Search Pipeline", executor=executor)

    pipeline.add(dataframe)
    pipeline.add(samples, X=dataframe)

    for train, persist in zip(train_datasets, persist_train):
        pipeline.add(train, X=samples)
        pipeline.add(persist, X=train)
    pipeline.add(test_dataset, X=samples)
    pipeline.add(persist_test, X=test_dataset)

    for i in range(4):
        for features_id in features.keys():
            dataset_feature = features[features_id][i]
            pipeline.add(dataset_feature, X=persist_train[i])
            dataset_label = labels[f"{attr}-{features_id}"][i]
            model = models[f"{attr}-{features_id}"][i]
            metric = metrics[f"{attr}-{features_id}"][i]
            pipeline.add(dataset_label, X=persist_train[i])
            pipeline.add(model, X=dataset_feature, y=dataset_label)
            pipeline.add(metric, model=model, dataset=persist_test)
    
    r2_values = {}
    for key in metrics.keys():
        for index, r2 in enumerate(metrics[key]):
            r2_values[f"{key}_{index}"] = r2

    pipeline.add(dump, **r2_values)

    if pipeline_save_location is not None:
        pipeline._dag_g.render(outfile=pipeline_save_location, cleanup=True)

    return pipeline


def model_train(
    data, attribute, feature_sets, model_name, model_kwargs, address, fig_pipeline, csv_name, ret
):
    executor = create_executor(address)
    executor.client.upload_file("utils.py")

    print("Creating pipeline...")
    pipeline = create_pipeline(
        executor, attribute, data, feature_sets, model_name, model_kwargs, fig_pipeline, csv_name
    )

    print("Executing pipeline...")
    start = perf_counter()
    pipeline.run()
    end = perf_counter()

    exec_time = end - start
    print(f"Done! Execution time: {exec_time:.2f} s")
    ret.value = exec_time


def model_search(args):
    start = perf_counter()
    with open(args.search_space, "r") as f:
        search_space = json.load(f)
    models_configs = {}
    feature_sets = search_space["features"]
    hiperparameters_sets = itertools.product(
        *list(search_space["hiperparameters"].values())
    )
    hiperparameters = list(search_space["hiperparameters"].keys())
    for i, h_set in enumerate(hiperparameters_sets):
        models_configs[f"model_{i}"] = {k: v for k, v in zip(hiperparameters, h_set)}
    print(
        f"Exploring {len(models_configs)} different hiperparameters configurations with {len(feature_sets)} feature sets! {len(feature_sets)*len(models_configs)} total configs!"
    )
    with open(f"{args.output}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model_name",
                "label",
                "x",
                "y",
                "z",
                "r2_0",
                "r2_1",
                "r2_2",
                "r2_3",
            ]
        )
    with open(f"{args.output}.json", "w") as f:
        f.write(json.dumps(models_configs, indent=4))

    ret = Value("d", 0)
    with open(f"{args.output}_times.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["model_name", "attribute", "pipeline_time"])
    for attribute in ["env", "cos", "freq"]:
        for model_name, model_kwargs in tqdm(models_configs.items()):
            p = Process(
                target=model_train,
                args=(
                    args.data,
                    attribute,
                    feature_sets,
                    model_name,
                    model_kwargs,
                    args.address,
                    args.fig_pipeline,
                    f"{args.output}.csv",
                    ret
                ),
            )
            p.start()
            p.join()
            with open(f"{args.output}_times.csv", "a") as f:
                writer = csv.writer(f)
                writer.writerow([model_name, attribute, ret.value])
            ret.value = 0

    end = perf_counter()
    print(f"Model search took {end - start:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--data",
        help="path to input dataframe",
        type=Path,
        default="features.parquet",
    )

    parser.add_argument(
        "-s",
        "--search-space",
        help="json file defining search space",
        type=Path,
        default="model_tuning.json",
    )

    parser.add_argument(
        "-e",
        "--address",
        help="Dask Scheduler address HOST:PORT",
        type=str,
        default=None,
    )

    parser.add_argument(
        "-f",
        "--fig-pipeline",
        help="name of file to save pipeline figure, only last one persists",
        type=str,
        default=None,
    )

    parser.add_argument(
        "-o",
        "--output",
        help="base name to save results",
        type=str,
        default="search_results",
    )

    args = parser.parse_args()

    model_search(args)
