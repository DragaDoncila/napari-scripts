from tqdm import tqdm
import pandas as pd
import numpy as np
import zarr
import time

CHUNK_SIZES = [
    256,
    512,
    1024,
    2048,
    4096,
    10980
]
FILENAMES = [f"Chunk_{chunk_size}_55HBU.zarr" for chunk_size in CHUNK_SIZES]
CHUNK_DATA = zip(FILENAMES, CHUNK_SIZES)

times = []
pbar = tqdm(CHUNK_DATA, total=len(FILENAMES))

for fn, chunk in pbar:
    slices_zarr = zarr.open(fn, mode='r')
    convert_start = time.time()
    slice_np = slices_zarr[0]
    convert_end = time.time()

    convert_time = convert_end - convert_start
    times.append(convert_time)
    pbar.set_description(f"Processing {chunk}")

df = pd.DataFrame({'filename' : FILENAMES, 'chunk_size': CHUNK_SIZES, 'time' : times})
df.to_csv('ReadTimes_ChunkSizes.csv')