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

def filter_path(pth):
    return pth.name.endswith('.tif') and not pth.name.endswith('ATB_R1.tif') and not pth.name.endswith('ATB_R2.tif')

# declare path to directory containing one tile
HOME_PATH = "/media/draga/My Passport/pepsL2A_zip_img/55HBU/"
NUM_IMS = 20 # images per tile per timestamp

# get all timestamps for each tile
all_dirs = os.listdir(HOME_PATH)
timestamps = []
for fl in all_dirs:
    if fl.endswith('.zip'):
        timestamp = fl.split('_')[1]
        timestamps.append([timestamp, fl])


timestamps.sort(key= lambda x: x[0])
grand_dask = [[None for j in range(2)] for i in range(NUM_IMS)]
i = len(timestamps)
tiff_shape = (0,0)
tiff_dtype = None
for timestamp, fn in timestamps:
    f_zip = zipfile.ZipFile(HOME_PATH + fn)

    f_path = zipfile.Path(f_zip)
    # get all the tif paths for this timestamp
    for dr in f_path.iterdir():
        tiff_drs = filter(filter_path, dr.iterdir())
        tiff_pths = sorted([tiff_pth.open() for tiff_pth in tiff_drs],
        key= lambda pth: pth.name)
    
    # open each tiff in this timestamp and append it to the grand array
    for j, tiff_pth in enumerate(tiff_pths):
        tiff_f = tifffile.TiffFile(tiff_pth)
        tiff_shape = tiff_f.pages[0].shape
        tiff_dtype = tiff_f.pages[0].dtype

        grand_dask[j][0] = tiff_shape
        grand_dask[j][1] = tiff_dtype

        tiff_delayed = dask.delayed(delay_as_array)(tiff_f)
        grand_dask[j].append(tiff_delayed)

        print(tiff_pth.name, tiff_shape, tiff_dtype)
        tiff_f.close()

    i -=1
    print("{} timestamps remaining".format(i))

    if i == 105:
        break;

all_images = []
# for each image of this resolution
for im_type in grand_dask:
    all_images.append(
        da.stack(
            [da.from_delayed(im, shape=im_type[0], dtype=im_type[1]) for im in im_type[2:]]
            )
        )

for image in all_images:
    print("type {}\nshape {}\nchunks {}\n".format(type(image), image.shape, image.chunks))

with napari.gui_qt():
    v = napari.view_image(all_images[0], name='Im 1', is_pyramid=False)

    for i, img in enumerate(all_images[1:]):
        v.add_image(img, name="Im " + str(i+1), is_pyramid=False, visible=False)

