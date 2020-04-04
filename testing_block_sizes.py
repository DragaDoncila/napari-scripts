import zarr
import time
import numpy as np
import tifffile
import pandas as pd
from numcodecs import Blosc

# HOME_PATH = "/media/draga/My Passport/pepsL2A_processed_img/55HBU/55HBU_Image.tif"

# tiff_f = tifffile.TiffFile(HOME_PATH)
# print(tiff_f.pages[0].dtype)
# tiff_f.close()

NUM_TIMEPOINTS = 5

CHUNK_SIZES = [
    256,
    512,
    1024,
    2048,
    4096,
    10980
]

FILENAMES = [f"Chunk_{chunk_size}_55HBU.zarr" for chunk_size in CHUNK_SIZES]

compressor = Blosc(
                cname='zstd', 
                clevel=9, 
                shuffle=Blosc.SHUFFLE, 
                blocksize=0
            )

zarr_im = zarr.open('/media/draga/My Passport/pepsL2A_zarr_processed/55HBU_Image.zarr', mode='r')
im_slice = zarr_im[:NUM_TIMEPOINTS, :, :]

times = []

for fn, chunk in zip(FILENAMES, CHUNK_SIZES):
    
    slice_zarr = zarr.open(fn, 
                mode='w', 
                shape=im_slice.shape, 
                dtype=im_slice.dtype,
                chunks=(1, chunk, chunk), 
                compressor=compressor
            )
    slice_zarr[:, :, :] = im_slice[:, :, :]
    slice_zarr.close()

    convert_start = time.time()
    slice_zarr = zarr.open(fn, mode='r')
    slice_np = np.array(slice_zarr)
    convert_end = time.time()

    times.append(convert_end - convert_start)

df = pd.DataFrame({'filename' : sizes, 'chunk_size': CHUNK_SIZES, 'time' : times})
df.to_csv('ReadTimes_ChunkSizes.csv')