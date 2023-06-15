import zarr
import numpy as np
# import dask.dataframe as ddf
import pandas as pd

from tqdm import tqdm

data = zarr.load("data/F3_train.zarr")

x = 2
y = 2 
z = 2
def feature_name(dim, pos):
    name = ["i", "j", "k"]
    name[dim] = f"{name[dim]}{'+' if pos > 0 else '-'}{np.abs(pos)}"
    return f"({','.join(name)})"

data = np.pad(data, mode="edge", pad_width=2)

data_2 = np.zeros((data.shape[0], data.shape[1], data.shape[2], 2*(x+y+z)+1))

for i in tqdm(list(range(z, data.shape[0]-z))):
    for j in range(y, data.shape[1]-y):
        for k in range(x, data.shape[2]-x):
            c = 0
            data_2[i][j][k][c] =  data[i][j][k]
            for off in range(-x, x+1):
                if off == 0:
                    continue
                c += 1
                data_2[i][j][k][c] =  data[i][j][k+off]
            for off in range(-y, y+1):
                if off == 0:
                    continue
                c += 1
                data_2[i][j][k][c] =  data[i][j+off][k]
            for off in range(-z, z+1):
                if off == 0:
                    continue
                c += 1
                data_2[i][j][k][c] =  data[i+off][j][k]
columns = ["(i,j,k)"]
for dim, n in zip([2, 1, 0], [x, y, z]):
    for pos in range(-n, n + 1):
        if pos == 0:
            continue
        columns.append(feature_name(dim, pos))

data_2 = data_2[z:-z,y:-y,x:-x]

df = pd.DataFrame(data_2.reshape(-1, data_2.shape[-1]), columns=columns)
df.to_parquet("default.parquet")
