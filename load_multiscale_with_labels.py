import numpy as np
import zarr
import napari
import pandas as pd
from ast import literal_eval
import tifffile


FN = "/media/draga/My Passport/Zarr/55HBU_Multiscale_Zarr_SingleChannel.zarr"
FNL = "../55HBUmap_TempCNN.tif"
FNC = "../55HBUconfmap_TempCNN.tif"
LABEL_MAPPING = "../class_map.txt"

if __name__ == "__main__":
    ims = []
    for i in range(6):
        im = zarr.open(FN + f"/{i}", "r")
        ims.append(im)

    df = pd.read_csv(LABEL_MAPPING)

    dicts = df.to_dict('split')
    classes = list(df['class'])
    colors = [tuple([v / 255 for v in literal_eval(val)]) for val in list(df['colour'])]

    label_properties = {
        'class': ['None'] + classes
    }

    colour_indices = [i for i in range(df.shape[0] + 1)]
    colours = [(0, 0, 0, 0)] + colors
    colour_dict = dict(zip(colour_indices, colours))
    labels = tifffile.TiffFile(FNL).pages[0].asarray()

    uncertainty = tifffile.TiffFile(FNC).pages[0].asarray()
    uncertainty = 1 - uncertainty

    with napari.gui_qt():
        v = napari.Viewer()
        v.add_image(ims, multiscale=True)
        v.add_labels(
            labels, 
            multiscale=False,
            properties=label_properties,
            color=colour_dict)
        v.add_image(
            uncertainty,
            colormap='magma'
        )