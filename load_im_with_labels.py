import napari
import zarr
from glob import glob
import time
import tifffile

im = zarr.open('../55HBU_GapFilled_Image.zarr', mode = 'r')
label_im = tifffile.TiffFile("../55HBUmap_TempCNN.tif")

with napari.gui_qt():
    v = napari.Viewer()
    v.add_image(
        im, 
        multiscale=False,
        name="Masked"
        )
    v.add_labels(
        label_im.pages[0].asarray(), 
        multiscale=False,
        name="Labels"
        )