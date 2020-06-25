import napari
import zarr
from skimage.transform import pyramid_gaussian
import numpy as np

im = zarr.open('../55HBU_GapFilled_Image.zarr', mode = 'r')

# mins = np.empty(im.shape[0], dtype=im.dtype)
# maxs = np.empty(im.shape[0], dtype=im.dtype)
# for i in range(im.shape[0]):
#     current_min = np.amin(im[i, :, :])
#     current_max = np.amax(im[i, :, :])
#     mins[i] = current_min
#     maxs[i] = current_max

# print(np.min(mins)) # 0
# print(np.max(maxs)) # 19254


im_slice = im[4, :, :]
im_pyramid = list(pyramid_gaussian(im_slice, downscale=2))
im_pyramid = [im_slice] + im_pyramid
with napari.gui_qt():
    v = napari.Viewer() 
    v.add_image(im_pyramid, multiscale=True)
    v.add_image(im_slice, multiscale=False)