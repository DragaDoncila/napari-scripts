import napari
import zarr
import skimage
from skimage.transform import pyramid_gaussian
import numpy as np
from numcodecs import Blosc
from pathlib import Path
import functools
import operator
from tqdm import tqdm
import json
import dask.array as da

MAX_LAYER = 5
DOWNSCALE = 2
FILENAME = '../55HBU_GapFilled_Image.zarr'
OUTDIR = "/media/draga/My Passport/Zarr/55HBU_Multiscale_Zarr_SingleChannel.zarr"
CHUNKSIZE = 1024
NEEDED_CHANNELS = 1
NUM_CHANNELS = 10

im = da.from_zarr(FILENAME)

im_shape = im.shape
num_slices = im_shape[0] // NUM_CHANNELS
x = im_shape[1]
y = im_shape[2]
im = da.reshape(im, (num_slices, NUM_CHANNELS, x, y))
compressor = Blosc(cname='zstd', clevel=9, shuffle=Blosc.SHUFFLE, blocksize=0)

Path(OUTDIR).mkdir(parents=True, exist_ok=True)
# open zarr arrays for each resolution shape (num_slices, res, res)
zarrs = []
for i in range(MAX_LAYER+1):
    new_res = tuple(np.ceil(np.array(im_shape[1:]) / (DOWNSCALE ** i)).astype(int)) if i != 0 else im_shape[1:]
    outname = OUTDIR + f"/{i}"

    z_arr = zarr.open(
            outname, 
            mode='w', 
            shape=(num_slices, NEEDED_CHANNELS, 1, new_res[0], new_res[1]), 
            dtype=im.dtype,
            chunks=(1, 1, 1, CHUNKSIZE, CHUNKSIZE), 
            compressor=compressor
            )
    zarrs.append(z_arr)

# # array of size 2**16 for frequency counts
contrast_histogram = functools.reduce(operator.add, (np.bincount(arr.ravel(), minlength=2**16) for arr in im), np.zeros(2**16))
# # for each slice
for i in tqdm(range(num_slices)):
    for j in tqdm(range(NEEDED_CHANNELS)):
        im_slice = im[i, j+1, :, :]
        # get pyramid
        im_pyramid = list(pyramid_gaussian(im_slice, max_layer=MAX_LAYER, downscale=DOWNSCALE))
        # for each resolution
        for k, new_im in enumerate(im_pyramid):
            print(k, i, j)
            # conver to uint16
            new_im = skimage.img_as_uint(new_im)
            # store into appropriate zarr at (slice, :, :)
            zarrs[k][i, j, 0, :, :] = new_im

# get 95th quantile of frequency counts
lower_contrast_limit = np.flatnonzero(np.cumsum(contrast_histogram) / np.sum(contrast_histogram) > 0.025)[0]
upper_contrast_limit = np.flatnonzero(np.cumsum(contrast_histogram) / np.sum(contrast_histogram) > 0.975)[0]

# write zattr file with contrast limits, whatever else it needs
zattr_dict = {}
zattr_dict["multiscales"] = []
zattr_dict["multiscales"].append({"datasets" : []})
for i in range(MAX_LAYER):
    zattr_dict["multiscales"][0]["datasets"].append({
        "path": f"{i}"
    })
zattr_dict["multiscales"][0]["version"] = "0.1"

zattr_dict["omero"] = {"channels" : []}
for i in range(NEEDED_CHANNELS):
    zattr_dict["omero"]["channels"].append(
        {
        "active" : i==0,
        "coefficient": 1,
        "color": "FFFFFF",
        "family": "linear",
        "inverted": "false",
        "label": str(i),
        "window": {
            "end": upper_contrast_limit,
            "max": 65535,
            "min": 0,
            "start": lower_contrast_limit
        }
        }
    )
zattr_dict["omero"]["id"] = str(0)
zattr_dict["omero"]["name"] = "55HBU"
zattr_dict["omero"]["rdefs"] = {
    "defaultT": 0,                    # First timepoint to show the user
    "defaultZ": 0,                  # First Z section to show the user
    "model": "color"                  # "color" or "greyscale"
}
zattr_dict["omero"]["version"] = "0.1"

with open(OUTDIR + "/.zattrs") as outfile:
    json.dump(zattr_dict, outfile)

with open(OUTDIR + "/.zgroups") as outfile:
    json.dump({"zarr_format": MAX_LAYER}, outfile)

# write histogram to file
np.savetxt("contrast_55HBU.csv", contrast_histogram, delimiter=",")
