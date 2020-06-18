import os
import tifffile
import zipfile
import shutil
import numpy as np
from skimage.io import imread, imsave
from glob import glob

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

SCALES = np.concatenate([np.ones((len(IM_SHAPES), 1)), 10980 * 10 / np.array(IM_SHAPES)], axis=1)  # 10m per pix is highest res
OFFSETS = [(5, 5) if shape[0] == 5490 else (0, 0) for shape in IM_SHAPES]
SHAPES = dict(zip(BANDS, IM_SHAPES))
OFFSETS = dict(zip(BANDS, OFFSETS))
SCALES = dict(zip(BANDS, SCALES))

CONTRAST_LIMITS = [-1000, 19_000]
QKL_SCALE = (1, 109.8, 109.8)

PATH = "/media/draga/My Passport/pepsL2A_zip_img/55HBU/"
OUT_PATH = "test_zips/55HBU_Corner2/"

in_zip_paths = sorted(glob(PATH + '/*.zip'))
out_zip_paths = [OUT_PATH + fname[:-4] for fname in [os.path.basename(zip_path) for zip_path in in_zip_paths]]

i = 0
for in_zip_path, out_zip_path in zip(in_zip_paths, out_zip_paths):
    print(out_zip_path)

    # open up in_zip
    in_zip = zipfile.ZipFile(in_zip_path, 'r')
    basepath = os.path.splitext(os.path.basename(in_zip_path))[0]
    
    # create out_dir
    out_dir = os.makedirs(out_zip_path + "/" + basepath)
    for band, shape in zip(BANDS, SHAPES):
        # open tif
        tiff_path = basepath + '/' + basepath + '_' + band + '.tif'
        tiff_file = in_zip.open(tiff_path)
        tiff_im = tifffile.TiffFile(tiff_file)
        
        new_tiff = np.zeros(tiff_im.pages[0].shape, dtype=tiff_im.pages[0].dtype)
        new_tiff[0:50, 0:50] = tiff_im.pages[0].asarray()[0:50, 0:50]

        # save as new tiff in out_zips (same name)
        out_path = out_zip_path + "/" + tiff_path
        imsave(out_path, new_tiff, compress=1)
        print("Saved:", out_path)

    # open quicklook
    jpg_path = basepath + '/' + basepath + '_' + 'QKL_ALL.jpg'
    jpg_file = in_zip.open(jpg_path)
    jpg_im = imread(jpg_file)
    new_im = np.zeros(jpg_im.shape, jpg_im.dtype)
    
    # slice quicklook
    new_im[0:50, 0:50, :] = jpg_im[0:50, 0:50, :]

    # save quicklook to dir
    out_path = out_zip_path + "/" + jpg_path
    imsave(out_path, new_im, compress=1)
    print("Saved:", out_path)

    #zip up directory
    zipf = zipfile.ZipFile(out_zip_path + ".zip", "w", zipfile.ZIP_DEFLATED)
    for root, dirs, filenames in os.walk(out_zip_path + "/" + basepath):
        for filename in filenames:
            filepath = os.path.join(root, filename)
            zipf.write(filepath, arcname=basepath + "/" + os.path.basename(filepath))
    zipf.close()

    shutil.rmtree(out_zip_path)
    i += 1

    if i == 10:
        break