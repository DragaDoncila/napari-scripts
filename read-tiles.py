import tifffile
import napari
import zipfile
import re
import os
import dask
import dask.array as da
import numpy as np


def delay_as_array(tiff_file_object):
    return tiff_file_object.pages[0].asarray()

# declare path to directory containing one tile
HOME_PATH = "/media/draga/My Passport/pepsL2A_zip_img/55HBU/"

# get all timestamps for each tile
all_dirs = os.listdir(HOME_PATH)
timestamps = []
for fl in all_dirs:
    if fl.endswith('.zip'):
        timestamp = fl.split('_')[1]
        timestamps.append([timestamp, fl])


timestamps.sort(key= lambda x: x[0])
grand_dask = []
i = len(timestamps)
tiff_shape = (0,0)
tiff_dtype = None
for timestamp, fn in timestamps:
    f_zip = zipfile.ZipFile(HOME_PATH + fn)
    tiff_zip = f_zip.open(fn[:-4] + '/' + fn[:-4] + '_SRE_B2.tif')
    tiff_f = tifffile.TiffFile(tiff_zip)

    tiff_delayed = dask.delayed(delay_as_array)(tiff_f)
    tiff_shape = tiff_f.pages[0].shape
    tiff_dtype = tiff_f.pages[0].dtype
    
    grand_dask.append(tiff_delayed)

    tiff_f.close()
    i -=1
    print("{} files remaining".format(i))

all_images = da.stack([da.from_delayed(del_array, shape=tiff_shape, dtype=tiff_dtype) for del_array in grand_dask])
print("type {}, shape {}, dtype {}, chunks {}".format(type(all_images), all_images.shape, all_images.dtype, all_images.chunks))
with napari.gui_qt():
    v = napari.view_image(all_images, name='SRE_B2', is_pyramid=False)

