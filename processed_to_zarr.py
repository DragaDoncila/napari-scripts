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
tiff_f.close()

d_transposed = d_mmap.transpose((2, 1, 0))
chunk_size = d_transposed.shape[1] // NUM_CHUNKS

info_file = open("/media/draga/My Passport/pepsL2A_zarr_processed/Compress_Info.txt", 'w')

for cmp_name in numcodecs.blosc.list_compressors():
    print("Trying {}".format(cmp_name))
    compressor = numcodecs.blosc.Blosc(cname=cmp_name, clevel=9)

    z_name = '/media/draga/My Passport/pepsL2A_zarr_processed/55HBU_Image_{}.zarr'.format(cmp_name)
    z_arr = zarr.open(z_name, mode='w', 
                    shape=(d_transposed.shape[0], chunk_size, d_transposed.shape[2]), 
                    dtype=d_transposed.dtype,
                    chunks=(1, None, None), compressor=compressor)

    start = 0
    end = 0
    for i in range(1):
        start = i * chunk_size
        end = start + chunk_size

        copy_start = time.time()
        print("Copying...")
        current_slice = np.copy(d_transposed[:,  start:end, :])
        copy_end = time.time()

        print("Copying complete: {} minutes, assigning...".format((copy_end - copy_start) / 60))
        z_arr[:, start:end, :] = current_slice[:, :, :]
        assign_end = time.time()
        print("Assigned: {} minutes".format((assign_end - copy_end) / 60))

    print(z_arr.info, file=info_file)
    print("#" * 20, end='\n\n', file=info_file)

    print(z_arr.info)
    print("#" * 20, end='\n\n')

info_file.close()

    # remainder that doesn't fit into an even chunk
    # z_arr[:, :, end:] = d_transposed[:, :, end:]