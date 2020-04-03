import tifffile
import zarr
import dask
import dask.array as da
import numpy as np
import time
from numcodecs import Blosc

HOME_PATH = "/media/draga/My Passport/pepsL2A_processed_img/55HBU/"
CHUNK_SIZE = 1

tiff_f = tifffile.TiffFile(HOME_PATH + "/55HBU_Image.tif")

d_mmap = tiff_f.pages[0].asarray(out='memmap')
tiff_f.close()

d_transposed = d_mmap.transpose((2, 1, 0))

compressor = Blosc(cname='zstd', clevel=9, shuffle=Blosc.SHUFFLE, blocksize=0)

z_name = '/media/draga/My Passport/pepsL2A_zarr_processed/55HBU_Image.zarr'
z_arr = zarr.open(z_name, mode='w', 
                shape=(d_transposed.shape[0], d_transposed.shape[1], d_transposed.shape[2]), 
                dtype=d_transposed.dtype,
                chunks=(1, None, None), compressor=compressor)

start = 0
end = 0

num_chunks = z_arr.shape[0] // CHUNK_SIZE
global_start = time.time()
#TODO: tqdm for progress bar?
for i in range(1):
    start = i * CHUNK_SIZE
    end = start + CHUNK_SIZE

    copy_start = time.time()
    print("Start: {}\tEnd: {}".format(start, end))
    print("Copying d_transposed[:, {}:{}, :]".format(start, end))
    current_slice = np.copy(d_transposed[start:end, :, :])
    copy_end = time.time()

    print("Copying complete: {} minutes.".format((copy_end - copy_start) / 60))
    print("Assigning slice into zarr..")
    z_arr[start:end, :, :] = current_slice
    assign_end = time.time()
    print("Assigned: {} minutes".format((assign_end - copy_end) / 60))

    print("{} chunks remaining...".format(num_chunks - i - 1))
    print("#*#" * 20)

# remainder zthat doesn't fit into an even chunk
print("Assigning remainder...")
z_arr[end:, :, :] = d_transposed[end:, :, :]

global_end = time.time()
print("TOTAL TIME: {}".format(global_end - global_start))