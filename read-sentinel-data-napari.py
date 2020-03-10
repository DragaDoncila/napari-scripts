# IPython log file


from skimage.external import tifffile
import napari
import zipfile
import re


#napari.view_image(arr, contrast_limits=[0, 2**16-1], is_pyramid=False)
f = zipfile.ZipFile('/media/draga/My Passport/pepsL2A_zip_img/55HDU/SENTINEL2A_20170720-002225-833_L2A_T55HDU_C_V1-0.zip')
zip_path = zipfile.Path(f)
tiffs_zip = []
for el in zip_path.iterdir():
    for el2 in el.iterdir():
        if not el2.is_dir() and '.tif' in el2.name:
            current_tiff = el2.open()
            tiffs_zip.append(current_tiff)

for tiff in tiffs_zip:
    print(tiff)
with napari.gui_qt():
    tiff_file = tifffile.TiffFile(tiffs_zip[0])
    v = napari.view_image(tiff_file.pages[0].asarray())
    for tiff in tiffs_zip[1:]:
        tiff_file = tifffile.TiffFile(tiff)
        v.add_image(tiff_file.pages[0].asarray())


# print(f.namelist())
# filenames = [fl.filename for fl in f.filelist]
# for filename in filenames:
#     reg_match = re.match('.*[?!/DATA/][?!/MASKS/].*\.tif', filename)
#     if reg_match:
#         print(filename)
# tiff_zip = f.open('SENTINEL2A_20170720-002225-833_L2A_T55HDU_C_V1-0/SENTINEL2A_20170720-002225-833_L2A_T55HDU_C_V1-0_FRE_B11.tif')
# tiff = tifffile.TiffFile(tiff_zip)
# tiff_zip2 = f.open('SENTINEL2A_20170720-002225-833_L2A_T55HDU_C_V1-0/SENTINEL2A_20170720-002225-833_L2A_T55HDU_C_V1-0_SRE_B2.tif')
# tiff2 = tifffile.TiffFile(tiff_zip2)
# with napari.gui_qt():
#     print(tiff.pages[0].shape)
#     print(tiff2.pages[0].shape)
#     v = napari.view_image(tiff.pages[0].asarray(), name='FRE_B11')
#     v.add_image(tiff.pages[0].asarray(), name='SRE_B2')
