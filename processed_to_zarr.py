import tifffile
import zarr
import dask
import dask.array as da
import numpy as np
import time
import numcodecs

HOME_PATH = "/media/draga/My Passport/pepsL2A_processed_img/55HBU/"
NUM_CHUNKS = 40

tiff_f = tifffile.TiffFile(HOME_PATH + "/55HBU_Image.tif")

d_mmap = np.memmap(HOME_PATH + "/55HBU_Image.tif", mode='r', dtype=tiff_f.pages[0].dtype, shape=tiff_f.pages[0].shape)
d_transposed = d_mmap.transpose((2, 1, 0))

tiff_f.close()

# check size of one chunk saved with different compressors
# must give the chunking parameter per axis
z_arr = zarr.open('/media/draga/My Passport/pepsL2A_zarr_processed/55HBU_Image.zarr', 
                mode='w', shape=d_transposed.shape, dtype=d_transposed.dtype,
                chunks=(1, None, None))

print(d_transposed.shape)
chunk_size = d_transposed.shape[1] // NUM_CHUNKS

start = 0
end = 0
for i in range(1):
    start = i * chunk_size
    end = start + chunk_size

    copy_start = time.time()
    print("Copying...")
    current_slice = np.copy(d_transposed[:,  start:end, :])
    copy_end = time.time()

    print("Copying complete: {} seconds, assigning...".format(copy_end - copy_start))
    z_arr[:, start:end, :] = current_slice[:, :, :]
    assign_end = time.time()
    print("Assigned: {} seconds".format(assign_end - copy_end))

# remainder that doesn't fit into an even chunk
# z_arr[:, :, end:] = d_transposed[:, :, end:]