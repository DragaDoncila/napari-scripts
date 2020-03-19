import tifffile
import zarr
import dask
import dask.array as da

def delay_as_array(tiff_file_object):
    return tiff_file_object.pages[0].asarray()

HOME_PATH = "/media/draga/My Passport/pepsL2A_processed_img/55HBU/"
NUM_CHUNKS = 40

tiff_f = tifffile.TiffFile(HOME_PATH + "/55HBU_Image.tif")

d_arr = da.from_delayed(dask.delayed(delay_as_array)(tiff_f), shape=tiff_f.pages[0].shape, dtype=tiff_f.pages[0].dtype)
d_transposed = d_arr.transpose((2, 0, 1))

z_arr = zarr.open('/media/draga/My Passport/pepsL2A_zarr_processed/55HBU_Image.zarr', 
                mode='w', shape=d_transposed.shape, dtype=d_transposed.dtype)

print(z_arr.chunks)

chunk_size = d_transposed.shape[2] // 40

start = 0
end = 0
for i in range(NUM_CHUNKS):
    start = i * chunk_size
    end = start + chunk_size

    z_arr[:, :, start:end] = d_transposed[:, :, start:end]

# remainder that doesn't fit into an even chunk
z_arr[:, :, end:] = d_transposed[:, :, end:]