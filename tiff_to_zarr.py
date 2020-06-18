"""Given a path to a folder containing subfolders of Sentinel zips and tiffs,
this script will produce a folder with identical structure and all images saved as
multiscale zarrs
"""

import tifffile
import zarr
import dask
import dask.array as da
import numpy as np
import time
from numcodecs import Blosc
from glob import glob
import os
import re
from pathlib import Path

HOME_PATH = "/media/draga/My Passport/"
CHUNK_SIZE = 30
OUT_PATH = "/media/draga/My Passport/Zarr/"

def convert_processed_to_zarr(filename, outname, chunk_size, axis_transpose):
    tiff_f = tifffile.TiffFile(filename)
    d_mmap = tiff_f.pages[0].asarray(out='memmap')
    tiff_f.close()
    d_transposed = d_mmap.transpose(axis_transpose)

    compressor = Blosc(cname='zstd', clevel=9, shuffle=Blosc.SHUFFLE, blocksize=0)

    z_arr = zarr.open(
                outname, 
                mode='a', 
                shape=(d_transposed.shape[0], d_transposed.shape[1], d_transposed.shape[2]), 
                dtype=d_transposed.dtype,
                chunks=(1, None, None), 
                compressor=compressor
                )

    start = 0
    end = 0
    num_chunks = z_arr.shape[0] // chunk_size

    global_start = time.time()
    #TODO: tqdm for progress bar?
    for i in range(0, 4):
        start = i * chunk_size
        end = start + chunk_size

        copy_start = time.time()
        print("Start: {}\tEnd: {}".format(start, end))
        print("Copying d_transposed[{}:{}, :, :]".format(start, end))
        current_slice = np.copy(d_transposed[start:end, :, :])
        copy_end = time.time()

        print("Copying complete: {} minutes.".format((copy_end - copy_start) / 60))
        print("Assigning slice into zarr...")
        z_arr[start:end, :, :] = current_slice
        assign_end = time.time()
        print("Assigned: {} minutes".format((assign_end - copy_end) / 60))

        del(current_slice)
        print("{} chunks remaining...".format(num_chunks - i - 1))
        print("#*#" * 20)


    if z_arr.shape[0] % chunk_size != 0:
        print("Copying remainder...")
        final_slice = np.copy(d_transposed[end:, :, :])
        print("Assigning remainder...")
        z_arr[end:, :, :] = final_slice

    global_end = time.time()
    print("TOTAL TIME: {}".format(global_end - global_start))

def get_paths_for_conversion(root_path, pattern):
    """Get list of absolute paths to all files within root_path and subdirectories which contain Sentinel images

    Args:
        root_path (str): path to directory containing Sentinel tiffs or subdirectories containing Sentinel tiffs

    Returns:
        List: list of absolute paths to all files containing Sentinel images within root_path
    """

    processed_path_list = []
    raw_path_list = []

    path_list = [x[0] for x in os.walk(root_path)]
    tile_paths = list(filter(
        lambda pth: re.search(pattern, pth),
        path_list))

    for tile_directory in tile_paths:
        # any directory that contains tiffs is considered a processed image
        processed_path_list.extend(glob(tile_directory + "/*.tif"))
        # any directory that contains zips is considered a raw image
        raw_path_list.extend(glob(tile_directory + "/*.zip"))

    return processed_path_list, raw_path_list

def get_in_out_mapping(paths, out_path, pattern):
    path_mapping = {}
    for pth in paths:
        match = re.search(pattern, pth)
        match_groups = match.groups()
        out_dir = out_path + "".join(match_groups[1:len(match_groups)-1])
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        path_mapping[pth] = out_dir
    return path_mapping


if __name__ == "__main__":
    pattern = rf"({HOME_PATH})(.*)([0-9][0-9][A-Z][A-Z][A-Z])$"
    processed_paths, raw_paths = get_paths_for_conversion(HOME_PATH, pattern)

    tif_split_pattern = rf"({HOME_PATH})(.*)([0-9][0-9][A-Z][A-Z][A-Z].*/)(.*.tif)"
    zip_split_pattern = rf"({HOME_PATH})(.*)([0-9][0-9][A-Z][A-Z][A-Z].*/)(.*.zip)"
    processed_path_mapping = get_in_out_mapping(processed_paths, OUT_PATH, tif_split_pattern)
    raw_path_mapping = get_in_out_mapping(raw_paths, OUT_PATH, zip_split_pattern)