import tifffile
from skimage.filters.rank import majority
from skimage.morphology import disk
from pathlib import Path
import zarr
import numpy as np
from numcodecs.blosc import Blosc

CHUNKSIZE = 1024
MAX_LAYERS = 5
OUTDIR = "/media/draga/My Passport/Zarr/55HBU_Multiscale_Labels.zarr"
Path(OUTDIR).mkdir(parents=True, exist_ok=True)
compressor = Blosc(cname='zstd', clevel=9, shuffle=Blosc.SHUFFLE, blocksize=0)

label_im = tifffile.TiffFile("../55HBUmap_TempCNN.tif").pages[0].asarray()
downsampled = label_im
for i in range(MAX_LAYERS+1):
    outname = OUTDIR + f"/{i}"
    z_arr = zarr.open(
        outname, 
        mode='w', 
        shape=downsampled.shape, 
        dtype=label_im.dtype,
        chunks=(CHUNKSIZE, CHUNKSIZE), 
        compressor=compressor
        )
    z_arr[:, :] = downsampled    

    label_im = majority(label_im, np.ones((2, 2)), shift_x=-1, shift_y=-1)
    downsampled = label_im[0::2, 0::2]

