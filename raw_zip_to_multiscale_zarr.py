import zarr
from skimage.transform import pyramid_gaussian
import numpy as np
from numcodecs import Blosc
from pathlib import Path
import functools
import operator
from tqdm import tqdm
import json
import dask.array as da
import dask
import tifffile
import zipfile
from glob import glob
import os
import skimage

MAX_LAYER = 5
DOWNSCALE = 2
CHUNKSIZE = 1024

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

def write_zattrs(contrast_limits, bands, outdir):
    # write zattr file with contrast limits and remaining attributes
    zattr_dict = {}
    zattr_dict["multiscales"] = []
    zattr_dict["multiscales"].append({"datasets" : []})
    for i in range(MAX_LAYER):
        zattr_dict["multiscales"][0]["datasets"].append({
            "path": f"{i}"
        })
    zattr_dict["multiscales"][0]["version"] = "0.1"

    zattr_dict["omero"] = {"channels" : []}
    for band in bands:
        lower_contrast_limit, upper_contrast_limit = contrast_limits[band]
        color = '0000FF' if band == 'SRE_B2' else 'green' if band=='00FF00' else 'FF0000' if band=='SRE_B4' else "FFFFFF"
        zattr_dict["omero"]["channels"].append(
            {
            "active" : band in {"SRE_B2", "SRE_B3","SRE_B4"},
            "coefficient": 1,
            "color": "FFFFFF",
            "family": "linear",
            "inverted": "false",
            "label": band,
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

    with open(outdir + "/.zattrs", "w") as outfile:
        json.dump(zattr_dict, outfile)

    with open(outdir + "/.zgroup", "w") as outfile:
        json.dump({"zarr_format": MAX_LAYER+1}, outfile)


# path to individual zip
DATA_ROOT_PATH = '/media/draga/My Passport/pepsL2A_zip_img/55HBU/'
OUTDIR = "/media/draga/My Passport/Zarr/55HBU_Raw"

# each zip file contains many bands, ie channels
BANDS_20M = [
    "FRE_B11",
    "FRE_B12",
    "FRE_B5",
    "FRE_B6",
    "FRE_B7",
    "FRE_B8A",
    "SRE_B11",
    "SRE_B12",
    "SRE_B5",
    "SRE_B6",
    "SRE_B7",
    "SRE_B8A",
    ]

BANDS_10M = [
    "FRE_B2",
    "FRE_B3",
    "FRE_B4",
    "FRE_B8",
    "SRE_B2", # surface reflectance, blue
    "SRE_B3", # surface reflectance, green
    "SRE_B4", # surface reflectance, red
    "SRE_B8",
]

IM_SHAPE_20M = (5490, 5490)
IM_SHAPE_10M = (10980, 10980)

CONTRAST_LIMITS=[-1000, 19_000]

# get all timestamps for this tile, and sort them
all_zips = sorted(glob(DATA_ROOT_PATH + '/*.zip'))
timestamps = [os.path.basename(fn).split('_')[1] for fn in all_zips]
num_timepoints = len(timestamps)

# open zarr of shape (timepoints, 1, bands, res, res) for 10980 and 5490 separately for each pyramid resolution
compressor = Blosc(cname='zstd', clevel=9, shuffle=Blosc.SHUFFLE, blocksize=0)
Path(OUTDIR).mkdir(parents=True, exist_ok=True)

zarrs10 = []
zarrs20 = []
for i in range(MAX_LAYER+1):
    # compute downscaled shape for both resolutions
    new_res10 = tuple(np.ceil(np.array(IM_SHAPE_10M) / (DOWNSCALE ** i)).astype(int)) if i != 0 else IM_SHAPE_10M
    new_res20 = tuple(np.ceil(np.array(IM_SHAPE_20M) / (DOWNSCALE ** i)).astype(int)) if i != 0 else IM_SHAPE_20M
    
    outname10 = OUTDIR + f"/10m_Res.zarr/{i}"
    outname20 = OUTDIR + f"/20m_Res.zarr/{i}"

    z_arr10 = zarr.open(
            outname10, 
            mode='w', 
            shape=(num_timepoints, len(BANDS_10M), 1, new_res10[0], new_res10[1]), 
            dtype=np.int16,
            chunks=(1, 1, 1, CHUNKSIZE, CHUNKSIZE), 
            compressor=compressor
            )
    zarrs10.append(z_arr10)

    z_arr20 = zarr.open(
        outname20, 
        mode='w', 
        shape=(num_timepoints, len(BANDS_20M), 1, new_res20[0], new_res20[1]), 
        dtype=np.int16,
        chunks=(1, 1, 1, CHUNKSIZE, CHUNKSIZE), 
        compressor=compressor
        )
    zarrs20.append(z_arr20)

# array of size 2**16 for frequency counts per band
# combined frequency count for B2, B3, B4 for SRE and FRE
contrast_histogram_10m = dict(zip(
    BANDS_10M,
    [np.zeros(2**16) for i in range(len(BANDS_10M))]
))

contrast_histogram_20m = dict(zip(
    BANDS_20M,
    [np.zeros(2**16) for i in range(len(BANDS_20M))]
))

# for each timepoint, band:
for i, timestamp in tqdm(enumerate(timestamps)):
    current_zip_fn = all_zips[i]
    # 10m bands
    for j, band in tqdm(enumerate(BANDS_10M)):
        # get pyramid
        basepath = os.path.splitext(os.path.basename(current_zip_fn))[0]
        path = basepath + '/' + basepath + '_' + band + '.tif'
        image = da.from_delayed(
            ziptiff2array(current_zip_fn, path), shape=IM_SHAPE_10M, dtype=np.int16
        )
        # add to frequency counts
        contrast_histogram_10m[band] += np.bincount(image.ravel(), minlength=2**16)
        
        im_pyramid = list(pyramid_gaussian(image, max_layer=MAX_LAYER, downscale=DOWNSCALE))
        # for each resolution:
        for k, downscaled in enumerate(im_pyramid):
            print(f"Res: {k}, Band: {j}, Timestamp: {i}")
            # convert back to int16
            downscaled = skimage.img_as_int(downscaled)
            # store into appropriate zarr
            zarrs10[k][i, j, 0, :, :] = downscaled

    # 20m bands
    for j, band in tqdm(enumerate(BANDS_20M)):
        # add to frequency counts

        # get pyramid
        basepath = os.path.splitext(os.path.basename(current_zip_fn))[0]
        path = basepath + '/' + basepath + '_' + band + '.tif'
        image = da.from_delayed(
            ziptiff2array(current_zip_fn, path), shape=IM_SHAPE_20M, dtype=np.int16
        )
        # add to frequency counts
        contrast_histogram_20m[band] += np.bincount(image.ravel(), minlength=2**16)
        im_pyramid = list(pyramid_gaussian(image, max_layer=MAX_LAYER, downscale=DOWNSCALE))
        # for each resolution:
        for k, downscaled in enumerate(im_pyramid):
            print(f"Res: {k}, Band: {j}, Timestamp: {i}")
            # convert back to int16
            downscaled = skimage.img_as_int(downscaled)
            # store into appropriate zarr
            zarrs20[k][i, j, 0, :, :] = downscaled
    
#compute contrast limits
contrast_limits_10m = {}
for band in BANDS_10M:
    # get 95th quantile of frequency counts
    lower_contrast_limit = np.flatnonzero(np.cumsum(contrast_histogram_10m[band]) / np.sum(contrast_histogram_10m[band]) > 0.025)[0]
    upper_contrast_limit = np.flatnonzero(np.cumsum(contrast_histogram_10m[band]) / np.sum(contrast_histogram_10m[band]) > 0.975)[0]

    contrast_limits_10m[band] = (lower_contrast_limit, upper_contrast_limit)

contrast_limits_20m = {}
for band in BANDS_20M:
    # get 95th quantile of frequency counts
    lower_contrast_limit = np.flatnonzero(np.cumsum(contrast_histogram_20m[band]) / np.sum(contrast_histogram_20m[band]) > 0.025)[0]
    upper_contrast_limit = np.flatnonzero(np.cumsum(contrast_histogram_20m[band]) / np.sum(contrast_histogram_20m[band]) > 0.975)[0]

    contrast_limits_20m[band] = (lower_contrast_limit, upper_contrast_limit)

# write zattrs
write_zattrs(contrast_limits_10m, BANDS_10M, OUTDIR + "/10m_Res.zarr/")
write_zattrs(contrast_limits_20m, BANDS_20M, OUTDIR + "/20m_Res.zarr/")