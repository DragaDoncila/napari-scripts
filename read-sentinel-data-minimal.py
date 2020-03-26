import pandas as pd
import os
import sys
import tifffile
import napari
import zipfile
import re
from glob import glob
import dask
import dask.array as da
import numpy as np
import time
from collections import defaultdict


@dask.delayed
def ziptiff2array(zip_filename, path_to_tiff):
    """Return a NumPy array from a TiffFile within a zip file.

    Parameters
    ----------
    zip_filename : str
        Path to the zip file containing the tiff.
    path_to_tiff : str
        The path to the TIFF file within the zip archive.

    Returns
    -------
    image : numpy array
        The output image.

    Notes
    -----
    This is a delayed function, so it actually returns a dask task. Call
    ``result.compute()`` or ``np.array(result)`` to instantiate the data.
    """
    with zipfile.ZipFile(zip_filename) as zipfile_obj:
        open_tiff_file = zipfile_obj.open(path_to_tiff)
        tiff_f = tifffile.TiffFile(open_tiff_file)
        image = tiff_f.pages[0].asarray()
    return image


# hard coding metadata to avoid opening tiffs
DATA_ROOT_PATH = "." if len(sys.argv) == 1 else sys.argv[1]

# each zip file contains many bands, ie channels
BANDS = [
    "FRE_B11",
    "FRE_B12",
    "FRE_B2",
    "FRE_B3",
    "FRE_B4",
    "FRE_B5",
    "FRE_B6",
    "FRE_B7",
    "FRE_B8",
    "FRE_B8A",
    "SRE_B11",
    "SRE_B12",
    "SRE_B2",  # surface reflectance, red
    "SRE_B3",  # surface reflectance, green
    "SRE_B4",  # surface reflectance, blue
    "SRE_B5",
    "SRE_B6",
    "SRE_B7",
    "SRE_B8",
    "SRE_B8A",
]
IM_SHAPES = [
    (5490, 5490),
    (5490, 5490),
    (10980, 10980),
    (10980, 10980),
    (10980, 10980),
    (5490, 5490),
    (5490, 5490),
    (5490, 5490),
    (10980, 10980),
    (5490, 5490),
    (5490, 5490),
    (5490, 5490),
    (10980, 10980),
    (10980, 10980),
    (10980, 10980),
    (5490, 5490),
    (5490, 5490),
    (5490, 5490),
    (10980, 10980),
    (5490, 5490),
]
SCALES = 10980 * 10 / np.array(IM_SHAPES)  # 10m per pix is highest res
OFFSETS = [(5, 5) if shape[0] == 5490 else (0, 0) for shape in IM_SHAPES]
SHAPES = dict(zip(BANDS, IM_SHAPES))
OFFSETS = dict(zip(BANDS, OFFSETS))
SCALES = dict(zip(BANDS, SCALES))

# get all timestamps for this tile, and sort them
all_zips = sorted(glob(DATA_ROOT_PATH + '/*.zip'))
print(all_zips)
print(DATA_ROOT_PATH)
timestamps = [os.path.basename(fn).split('_')[1] for fn in all_zips]

# stack all timepoints together for each band
images = {}
for band, shape in zip(BANDS, IM_SHAPES):
    stack = []
    for fn in all_zips:
        basepath = os.path.splitext(os.path.basename(fn))[0]
        path = basepath + '/' + basepath + '_' + band + '.tif'
        image = da.from_delayed(
            ziptiff2array(fn, path), shape=shape, dtype=np.int16
        )
        stack.append(image)
    images[band] = da.stack(stack)


colormaps = defaultdict(lambda: 'gray')
for band in BANDS:
    if band.endswith('B2'):
        colormaps[band] = 'red'
    elif band.endswith('B3'):
        colormaps[band] = 'green'
    elif band.endswith('B4'):
        colormaps[band] = 'blue'


with napari.gui_qt():
    v = napari.Viewer()
    times = []
    visibles = []
    for band, image in images.items():
        colormap = colormaps[band]
        blending = 'additive' if colormaps[band] != 'gray' else 'translucent'
        visible = (band in {'SRE_B2', 'SRE_B3', 'SRE_B4'})
        start = time.time()
        v.add_image(
            image,
            name=band,
            is_pyramid=False,
            scale=SCALES[band],
            translate=OFFSETS[band],
            colormap=colormap,
            blending=blending,
            visible=visible,
            contrast_limits=[-1000, 19_000],
        )
        times.append(time.time() - start)
        visibles.append(visible)
    sizes = np.prod(IM_SHAPES, axis=1)
    df = pd.DataFrame({'sizes' : sizes, 'times' : times, 'visible' : visibles})
    print(df)
