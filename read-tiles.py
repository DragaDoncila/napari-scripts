import tifffile
import napari
import zipfile
import re
import os
import dask
import dask.array as da
import numpy as np
import time


def delay_as_array(im, f_zip):
    im_tiff = f_zip.open(im)  

    tiff_f = tifffile.TiffFile(im_tiff)

    return tiff_f.pages[0].asarray()

HOME_PATH = "/media/draga/My Passport/pepsL2A_zip_img/55HBU/"
NUM_IMS = 20
IM_SUFFIXES = ['FRE_B11', 'FRE_B12', 'FRE_B2', 'FRE_B3', 'FRE_B4', 'FRE_B5', 'FRE_B6', 'FRE_B7', 'FRE_B8', 'FRE_B8A',
                'SRE_B11', 'SRE_B12', 'SRE_B2', 'SRE_B3', 'SRE_B4', 'SRE_B5', 'SRE_B6', 'SRE_B7', 'SRE_B8', 'SRE_B8A']
IM_SHAPES = [(5490, 5490), (5490, 5490), (10980, 10980), (10980, 10980), (10980, 10980), (5490, 5490), (5490, 5490), (5490, 5490), (10980, 10980), (5490, 5490), 
             (5490, 5490), (5490, 5490), (10980, 10980), (10980, 10980), (10980, 10980), (5490, 5490), (5490, 5490), (5490, 5490), (10980, 10980), (5490, 5490)]
IM_DTYPE = np.dtype('int16')

# get all timestamps for each tile
all_dirs = os.listdir(HOME_PATH)
timestamps = []
for fl in all_dirs:
    if fl.endswith('.zip'):
        timestamp = fl.split('_')[1]
        timestamps.append([timestamp, fl])


timestamps.sort(key= lambda x: x[0])
grand_dask = [[] for i in range(NUM_IMS)]
i = len(timestamps)

for timestamp, fn in timestamps:
    f_zip = zipfile.ZipFile(HOME_PATH + fn)
    
    # open each tiff in this timestamp and append it to the grand array
    for j, suffix in enumerate(IM_SUFFIXES):
        current_im = fn[:-4] + '/' + fn[:-4] + '_' + suffix + '.tif'

        tiff_delayed = dask.delayed(delay_as_array)(current_im, f_zip)
        grand_dask[j].append(tiff_delayed)

    i -=1
    print("{} timestamps remaining".format(i))

    if i == 106:
        break;

all_images = []
# for each image of this resolution
for i, im_type in enumerate(grand_dask):
    all_images.append(
        da.stack(
            [da.from_delayed(im, shape=IM_SHAPES[i], dtype=IM_DTYPE) for im in im_type]
            )
        )

with napari.gui_qt():
    start = time.time()
    v = napari.view_image(all_images[0], name='Im 1', is_pyramid=False)
    end_1 = time.time()

    for i, img in enumerate(all_images[1:]):
        v.add_image(img, name="Im " + str(i+2), is_pyramid=False, visible=False)
    end_2 = time.time()

    print("First image: ", end_1 - start)
    print("Second image: ", end_2 - end_1)

