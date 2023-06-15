import dask.array as da
import dask.dataframe as ddf
import pandas as pd
import numpy as np
import zarr
import dask
import itertools
from dasf.transforms import Transform
from dasf.datasets import Dataset
from dasf.pipeline.executors import DaskPipelineExecutor
from dasf.utils.decorators import task_handler


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
        pad = self.__pad_width(len(X.shape))
        X = da.pad(X, pad, mode="edge")
        X = self.__neighbours(X, da)
        slices = self.__pad_to_slice(pad)
        return X[
            slices[0][0] : slices[0][1],
            slices[1][0] : slices[1][1],
            slices[2][0] : slices[2][1],
        ]

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


class FeaturesJoin(Transform):
    def _lazy_transform_cpu(self, **features):
        return da.stack(features.values(), axis=-1)

    def _transform_cpu(self, **features):
        return np.stack(features.values(), axis=-1)

class CreateDataFrame(Transform):
    def __init__(self, fname = "test.parquet"):
        self.fname = fname

    def _shuffle(self, chunk):
        index = np.arange(chunk.shape[0])
        np.random.shuffle(index)
        return chunk[index]

    def _lazy_transform_cpu(self, **features):
        data =  da.stack(features.values(), axis=-1)
        data = data.reshape(-1, data.shape[-1])

        # rechunk and shuffle
        data = data.rechunk({0: 5_000_000, 1: -1})
        data = data.map_blocks(self._shuffle)
        
        df = ddf.from_dask_array(data, columns=features.keys())
        print("Writing to disk")
        df.to_parquet(self.fname)
        print("Reading")
        df = ddf.read_parquet(self.fname)
        return df


    def _transform_cpu(self, **features):
        data = np.stack(features.values(), axis=-1)
        data = data.reshape(-1, data.shape[-1])
        df = pd.DataFrame(data, columns=features.keys())
        return df



def chunks_merge(chunks):
    new_chunks = []
    for comb in itertools.product(*reversed(chunks)):
        new_chunks.append(np.product(comb))
    return tuple(new_chunks)

class ReshapeFeatures(Transform):
    
    def _lazy_transform_cpu(self, X):
        X = X.reshape(-1, X.shape[-1])
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

    def _split(self, chunk):
        return chunk[:-1]

    def _lazy_transform_cpu(self, X):
        return X[X.columns.difference(["label"])]

    def _transform_cpu(self, X):
        return X[X.columns.difference(["label"])]
    
class SplitLabel(Transform):
    def _split(self, chunk):
        return chunk[-1:]
    
    def _lazy_transform_cpu(self, X):
        return X[["label"]]

    def _transform_cpu(self, X):
        return X[["label"]]
    
class SaveModel(Transform):
    def __init__(self, fname: str):
        self._fname = fname

    def transform(self, model):
        model.save_model(self._fname)


class SaveResult(Transform):
    def __init__(self, fname: str):
        self._fname = fname

    def _lazy_transform_cpu(self, X, raw):
        X = X.reshape(raw.shape)
        zarr.save(self._fname, X.compute())

    def _transform_cpu(self, X, raw):
        X = X.reshape(raw.shape)
        zarr.save(self._fname, X)

class SaveIntermediate(Transform):
    def __init__(self, fname: str):
        self._fname = fname

    def _lazy_transform_cpu(self, X):
        X = X.rechunk()
        X.to_zarr(self._fname, overwrite=True)

    def _transform_cpu(self, X):
        zarr.save(self._fname, X)

class LoadIntermediate(Dataset):
    def __init__(self, fname: str, chunks={0:"auto", 1: -1}):
        super().__init__(name="intermediate")
        self._fname = fname
        self.chunks = chunks

    def _lazy_load_cpu(self, dep):
        return da.from_zarr(self._fname, chunks=self.chunks)

    def _load_cpu(self, dep):
        return zarr.load(self._fname)

    @task_handler
    def load(self, dep):
        ...

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
