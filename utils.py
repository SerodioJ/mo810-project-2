import dask.array as da
import dask.dataframe as ddf
import pandas as pd
import numpy as np
import json
import zarr
import csv
import itertools
from time import sleep, perf_counter
from threading import Thread

from sklearn.metrics import r2_score
from dasf.transforms import Transform
from dasf.datasets import Dataset
from dasf.pipeline.executors import DaskPipelineExecutor
from dasf.utils.decorators import task_handler
from dask.distributed import wait
from distributed import Future


class ZarrDataset(Dataset):
    def __init__(self, name: str, data_path: str, chunks={0: "auto", 1: "auto", 2: -1}):
        super().__init__(name=name)
        self.data_path = data_path
        self.chunks = chunks

    def _lazy_load_cpu(self):
        return da.from_zarr(self.data_path, chunks=self.chunks)

    def _load_cpu(self):
        return zarr.load(self.data_path)

    @task_handler
    def load(self):
        ...


class NeighbourValue(Transform):
    def __init__(self, dim: int, pos: int):
        self._dim = dim
        self._pos = pos

    def __pad_width(self, dims):
        paddings = []
        for i in range(dims):
            if i != self._dim:
                paddings.append((0, 0))
                continue
            if self._pos > 0:
                paddings.append((0, self._pos))
            else:
                paddings.append((np.abs(self._pos), 0))
        return paddings

    def __pad_to_slice(self, pad):
        slices = []
        for t in pad:
            sli = (
                t[0] if t[0] else None,
                -t[1] if t[1] else None,
            )
            slices.append(sli)
        return slices

    def __neighbours(self, X, xp):
        return xp.roll(X, shift=-self._pos, axis=self._dim)

    def _lazy_transform_cpu(self, X):
        depth = [0] * len(X.shape)
        depth[self._dim] = np.abs(self._pos)
        return X.map_overlap(
            self.__neighbours,
            xp=np,
            depth=tuple(depth),
            boundary="nearest",
            meta=np.array(()),
        )

    def _transform_cpu(self, X):
        pad = self.__pad_width(len(X.shape))
        X = np.pad(X, pad, mode="edge")
        X = self.__neighbours(X, np)
        slices = self.__pad_to_slice(pad)
        return X[
            slices[0][0] : slices[0][1],
            slices[1][0] : slices[1][1],
            slices[2][0] : slices[2][1],
        ]


class Neighbourhood(Transform):
    def __init__(self, x: int, y: int, z: int):
        self._x = x
        self._y = y
        self._z = z

    def __pad_width(self, dims):
        paddings = []
        for i in range(dims):
            if i != self._dim:
                paddings.append((0, 0))
                continue
            if self._pos > 0:
                paddings.append((0, self._pos))
            else:
                paddings.append((np.abs(self._pos), 0))
        return paddings

    def __pad_to_slice(self, pad):
        slices = []
        for t in pad:
            sli = (
                t[0] if t[0] else None,
                -t[1] if t[1] else None,
            )
            slices.append(sli)
        return slices

    def __neighbours(self, X, xp):
        neighbourhood = [X]
        for dim, n in zip([2, 1, 0], [self._x, self._y, self._z]):
            for pos in range(-n, n + 1):
                if pos == 0:
                    continue
                neighbourhood.append(xp.roll(X, shift=-pos, axis=dim))
        return xp.stack(neighbourhood, axis=-1)

    def _lazy_transform_cpu(self, X):
        print(X.chunks)
        print(X.shape)
        chunks = []
        neighbours = (self._z, self._y, self._x)
        for pad, dim in zip(neighbours, X.chunks):
            dim_chunks = []
            for chunk in dim:
                dim_chunks.append(chunk + 2 * pad)
            chunks.append(tuple(dim_chunks))

        features = (self._x + self._y + self._z) * 2 + 1
        X = X.map_overlap(
            self.__neighbours,
            xp=np,
            depth=(self._z, self._y, self._x),
            new_axis=[3],
            chunks=(*chunks, (features,)),
            boundary="nearest",
            meta=np.array(()),
        )
        print(X.chunks)
        print(X.shape)
        return X


class GetChunkDistribution(Transform):
    def __init__(self, client, json_name):
        self._client = client
        self._json_name = json_name

    def _lazy_transform_cpu(self, X):
        start_comp = perf_counter()
        X = X.persist()
        wait(X)
        time_comp = perf_counter() - start_comp
        chunks_per_worker = {
            w: len(chunks) for w, chunks in dict(self._client.has_what()).items()
        }
        data = {
            "comp_time": time_comp,
            "old_chunks": chunks_per_worker,
        }
        with open(self._json_name, "w") as f:
            f.write(json.dumps(data, indent=4))
        return X

    def _transform_cpu(self, X):
        return X


class RebalanceData(Transform):
    def __init__(self, client, json_name):
        self._client = client
        self._json_name = json_name

    def _lazy_transform_cpu(self, X):
        start_comp = perf_counter()
        X = X.persist()
        wait(X)
        time_comp = perf_counter() - start_comp
        chunks_per_worker_old = {
            w: len(chunks) for w, chunks in dict(self._client.has_what()).items()
        }
        start_transfer = perf_counter()
        rebalance(client=self._client)
        chunks_per_worker_new = {
            w: len(chunks) for w, chunks in dict(self._client.has_what()).items()
        }
        time_transfer = perf_counter() - start_transfer
        data = {
            "time_comp": time_comp,
            "time_transfer": time_transfer,
            "old_chunks": chunks_per_worker_old,
            "new_chunks": chunks_per_worker_new,
        }
        with open(self._json_name, "w") as f:
            f.write(json.dumps(data, indent=4))
        return X

    def _transform_cpu(self, X):
        return X


class FeaturesJoin(Transform):
    def _lazy_transform_cpu(self, **features):
        stacked = da.stack(features.values(), axis=-1)
        return stacked.reshape(-1, stacked.shape[-1])

    def _transform_cpu(self, **features):
        return np.stack(features.values(), axis=-1)


class Reshape(Transform):
    def _lazy_transform_cpu(self, X):
        return X.reshape(-1, X.shape[-1]).rechunk({0: 500_000, 1: -1})

    def _transform_cpu(self, X):
        return X.reshape(-1, X.shape[-1])


class Repartition(Transform):
    def __init__(self, npartitions):
        self._npartitions = npartitions

    def _lazy_transform_cpu(self, X):
        return X.repartition(npartitions=self._npartitions)

    def _transform_cpu(self, X):
        return X


class Concatenate(Transform):
    def _lazy_transform_cpu(self, X, y):
        y = y.reshape(*y.shape, 1)
        print(X.shape)
        print(y.shape)
        join = da.concatenate([X, y], axis=-1)
        join = join.rechunk({-1: -1})
        return join


class CreateDataFrame(Transform):
    def __init__(self, fname="test.parquet", workers=None):
        self.workers = workers
        self.fname = fname

    def _shuffle(self, chunk):
        index = np.arange(chunk.shape[0])
        np.random.shuffle(index)
        return chunk[index]

    def _reshape(self, chunk):
        return chunk.reshape(-1, chunks.shape[-1])

    def _stack_reshape(self, *features):
        print(features)
        stacked_data = np.stack(*features, axis=-1)
        return stacked_data.reshape(-1, stacked_data.shape[-1])

    def _lazy_transform_cpu(self, **features):
        # mapping = []
        # f = list(features.values())[0]
        # for feature in features.values():
        #     mapping.extend([feature, "ijk"])
        # data = da.blockwise(self._stack_reshape, "xz", *mapping,
        #     dtype=f.dtype,
        #     new_axes={"x":np.product(f.shape),"z": len(features.values())}

        # )
        data = da.stack(features.values(), axis=-1)

        # print(data.chunks)
        # data = data.rechunk(({3: -1}))
        # print(data.chunks)
        # chunks = chunks_merge(data.chunks[:-1])
        # chunks = tuple([np.prod(chunk) for chunk in chunks])
        # print(chunks)
        # print(data.shape)
        data = data.reshape(-1, data.shape[-1])
        # print((chunks, (data.shape[-1],)))
        # data = data.map_blocks(self._reshape, dtype=data.dtype,
        #     drop_axis=[0, 1, 2], new_axis=[0], chunks=(chunks, (data.shape[-1],)))
        # print(data.chunks)
        # return data
        # f = list(features.values())[0]
        # chunks = chunks_merge(f.chunks)
        # print(f.chunks)
        # print(f.chunksize)
        # chunks = tuple([np.prod(chunk) for chunk in chunks])
        # data = da.map_blocks(self._stack_reshape, *features.values(), dtype=f.dtype,
        #     drop_axis=[0, 1, 2],
        #     new_axis=[0, 1],
        #     chunks=(chunks, (len(features.values()),)),
        #     meta=np.array(())
        # )
        # print(data.chunks)
        # return data

        # rechunk and shuffle
        # chunks = 5_000_000
        # if self.workers:
        #     chunks = (data.shape[0]//(self.workers*10)) + 1
        data = data.rechunk({0: "auto", 1: -1})
        return data
        # data = data.map_blocks(self._shuffle)

        df = ddf.from_dask_array(data, columns=features.keys())
        # print("Writing to disk")
        # df.to_parquet(self.fname)
        # print("Reading")
        # df = ddf.read_parquet(self.fname)
        return df

    def _transform_cpu(self, **features):
        data = np.stack(features.values(), axis=-1)
        data = data.reshape(-1, data.shape[-1])
        df = pd.DataFrame(data, columns=features.keys())
        return df


def chunks_merge(chunks):
    new_chunks = []
    for comb in itertools.product(*chunks):
        new_chunks.append(np.product(comb))
    return tuple(new_chunks)


class ReshapeFeatures(Transform):
    def _lazy_transform_cpu(self, X):
        X = X.reshape(-1, X.shape[-1], limit="2GB")
        X = X.rechunk({0: "auto", 1: -1})
        return X

    def _transform_cpu(self, X):
        return X.reshape(-1, X.shape[-1])


class ReshapeLabels(Transform):
    def _lazy_transform_cpu(self, X, y):
        y = y.flatten()
        y = y.rechunk(X.chunks[:-1])
        return y

    def _transform_cpu(self, X, y):
        return y.flatten()


class SplitFeatures(Transform):
    def __init__(self, feature_set=None):
        self._feature_set = feature_set

    def _lazy_transform_cpu(self, X):
        return X[self._feature_set]

    def _transform_cpu(self, X):
        return X[self._feature_set]


class SplitLabel(Transform):
    def __init__(self, label="label"):
        self._label = label

    def _lazy_transform_cpu(self, X):
        return X[[self._label]]

    def _transform_cpu(self, X):
        return X[[self._label]]


class SaveModel(Transform):
    def __init__(self, fname: str):
        self._fname = fname

    def transform(self, model):
        model.save_model(self._fname)


class SaveResult(Transform):
    def __init__(self, fname: str):
        self._fname = fname

    def _lazy_transform_cpu(self, X, raw):
        X = X.to_dask_array().compute_chunk_sizes()
        X = X.reshape(raw.shape)
        start = perf_counter()
        X = X.compute()
        end = perf_counter() - start
        zarr.save(self._fname, X)
        return end

    def _transform_cpu(self, X, raw):
        X = X.reshape(raw.shape)
        zarr.save(self._fname, X)


class LoadDataFrame(Dataset):
    def __init__(self, fname: str):
        super().__init__(name="dataframe")
        self._fname = fname

    def _lazy_load_cpu(self):
        return ddf.read_parquet(self._fname)

    def _load_cpu(self):
        return pd.read_parquet(self._fname)

    @task_handler
    def load(self):
        ...


class SampleDataframe(Transform):
    def __init__(self, split, random_state=10):
        self._split = split
        self._random_state = random_state

    def _lazy_transform_cpu(self, X):
        return X.random_split(self._split, random_state=self._random_state)


class TrainDataset(Transform):
    def __init__(self, index):
        self._index = index

    def _lazy_transform_cpu(self, X):
        return X[self._index]

    def _transform_cpu(self, X):
        return X[self._index]


class TestDataset(Transform):
    def __init__(self, index):
        self._index = index

    def _lazy_transform_cpu(self, X):
        return X[self._index]

    def _transform_cpu(self, X):
        return X[self._index]


class EvaluateModel(Transform):
    def __init__(self, feature_set, label):
        self._feature_set = feature_set
        self._label = label

    def transform(self, model, dataset=None):
        y_pred = model.predict(dataset[self._feature_set])
        return r2_score(dataset[[self._label]], y_pred)


class DumpToCSV(Transform):
    def __init__(self, fname, model_name, keys, num):
        self._fname = fname
        self._model_name = model_name
        self._keys = keys
        self._num = num

    def transform(self, **r2_scores):
        with open(self._fname, "a") as f:
            writer = csv.writer(f)
            for k in self._keys:
                label, x, y, z = k.split("-")
                writer.writerow(
                    [
                        self._model_name,
                        label,
                        x,
                        y,
                        z,
                        *[r2_scores[f"{k}_{i}"] for i in range(self._num)],
                    ]
                )


class SaveParquet(Transform):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    @staticmethod
    def feature_name(dim, pos):
        name = ["i", "j", "k"]
        name[dim] = f"{name[dim]}{'+' if pos > 0 else '-'}{np.abs(pos)}"
        return f"({','.join(name)})"

    def _lazy_transform_cpu(self, X):
        columns = ["(i,j,k)"]
        for dim, n in zip([2, 1, 0], [self.x, self.y, self.z]):
            for pos in range(-n, n + 1):
                if pos == 0:
                    continue
                columns.append(self.feature_name(dim, pos))
        df = ddf.from_dask_array(X, columns=columns)
        df.to_parquet("a.parquet")


def generate_neighbourhood_features(
    inline_window, trace_window, samples_window
) -> dict:
    def feature_name(dim, pos):
        name = ["i", "j", "k"]
        name[dim] = f"{name[dim]}{'+' if pos > 0 else '-'}{np.abs(pos)}"
        return f"({','.join(name)})"

    operators = {}
    for dim, n in zip([2, 1, 0], [samples_window, trace_window, inline_window]):
        for pos in range(-n, n + 1):
            if pos == 0:
                continue
            operators[feature_name(dim, pos)] = NeighbourValue(dim=dim, pos=pos)
    return operators


def create_executor(address: str = None) -> DaskPipelineExecutor:
    if address is not None:
        addr = ":".join(address.split(":")[:2])
        port = str(address.split(":")[-1])
        print(f"Creating executor. Address: {addr}, port: {port}")
        return DaskPipelineExecutor(local=False, use_gpu=False, address=addr, port=port)
    else:
        return DaskPipelineExecutor(local=True, use_gpu=False)


def rebalance(client):
    def join_exceeding_replicas(replicas_dict):
        replicas = []
        for replica_list in replicas_dict.values():
            replicas.extend(replica_list)
        return replicas

    def replicate(client, worker, futures):
        client.replicate(list(map(Future, futures)), workers=[worker])

    distribution = dict(client.has_what())
    total_chunks = sum(map(len, distribution.values()))
    expected_replicas_per_worker = total_chunks // len(distribution)
    src_workers = {}
    dst_workers = {}
    replica_ops = {}
    for worker, stored_keys in distribution.items():
        if len(stored_keys) > expected_replicas_per_worker:
            src_workers[worker] = stored_keys[expected_replicas_per_worker:]
        else:
            dst_workers[worker] = (expected_replicas_per_worker - len(stored_keys), [])
    exceeding_replicas = join_exceeding_replicas(src_workers)
    for dst, receive in dst_workers.items():
        replicas_missing, replicas = receive
        replica_ops[dst] = exceeding_replicas[:replicas_missing]
        exceeding_replicas = exceeding_replicas[replicas_missing:]
        if exceeding_replicas == []:
            break
    client.amm.stop()
    thread_pool = []
    for worker, futures in replica_ops.items():
        thread = Thread(target=replicate, args=(client, worker, futures))
        thread.start()
        thread_pool.append(thread)
    for thread in thread_pool:
        thread.join()
    client.amm.start()

    while sum(map(len, dict(client.has_what()).values())) != total_chunks:
        sleep(0.5)
