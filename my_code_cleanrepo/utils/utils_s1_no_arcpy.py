import os
from typing import Optional
import numpy as np
from osgeo import gdal
import re


def split_raw_s1_gdal(
    in_vv_tif_path:str,
    out_folder_path:str,
    out_pixel_size:int,
    out_tile_size:int,
    overlap:int,
    landmask_path:str=r'c:\Users\juvad3723\.1\--Data\land_mask\world_landmask_2500m.tif',
    custom_tile_name: Optional[str] = None,
    in_nodata_value: Optional[int]=0,
    ):
    # -----------------------------
    # STEP 1: WARP USING GCPs → EPSG:4326
    # -----------------------------
    src = gdal.Open(in_vv_tif_path)

    wgs84_ds = gdal.Warp(
        "",
        src,
        format="MEM",
        dstSRS="EPSG:4326",
        resampleAlg="bilinear"
    )

    gt = wgs84_ds.GetGeoTransform() 

    minx = gt[0]
    maxy = gt[3]
    maxx = gt[0] + gt[1] * wgs84_ds.RasterXSize
    miny = gt[3] + gt[5] * wgs84_ds.RasterYSize
    # -----------------------------
    # STEP 2: ALIGN LANDMASK
    # -----------------------------
    landmask = gdal.Open(landmask_path)

    landmask_aligned_wgs84 = gdal.Warp(
        "",
        landmask,
        format="MEM",
        dstSRS="EPSG:4326",
        width=wgs84_ds.RasterXSize,
        height=wgs84_ds.RasterYSize,
        outputBounds=[minx, miny, maxx, maxy],
        resampleAlg="nearest" # we can use nearest resampling because binary landmask
    )

    # -----------------------------
    # STEP 3: APPLY LANDMASK
    # -----------------------------
    img = wgs84_ds.ReadAsArray()
    mask = landmask_aligned_wgs84.ReadAsArray()

    masked = np.where(mask == 1, img, 0)

    # Write back to MEM dataset
    driver = gdal.GetDriverByName("MEM")
    masked_ds = driver.Create(
        "",#no name for memory
        wgs84_ds.RasterXSize,
        wgs84_ds.RasterYSize,
        1,
        wgs84_ds.GetRasterBand(1).DataType
    )

    masked_ds.SetGeoTransform(wgs84_ds.GetGeoTransform())
    masked_ds.SetProjection(wgs84_ds.GetProjection())
    masked_ds.GetRasterBand(1).WriteArray(masked)

    # -----------------------------
    # STEP 4: COMPUTE UTM ZONE
    # -----------------------------
    gt = masked_ds.GetGeoTransform()

    center_lon = gt[0] + gt[1] * (masked_ds.RasterXSize / 2)
    center_lat = gt[3] + gt[5] * (masked_ds.RasterYSize / 2)

    zone = int((center_lon + 180) / 6) + 1
    epsg = 32600 + zone if center_lat >= 0 else 32700 + zone

    print(f"Using UTM EPSG:{epsg}")

    # -----------------------------
    # STEP 5: REPROJECT TO UTM
    # -----------------------------
    utm_ds = gdal.Warp(
        "",#no name for memory
        masked_ds,
        format="MEM",
        dstSRS=f"EPSG:{epsg}",
        xRes=out_pixel_size,
        yRes=out_pixel_size,
        resampleAlg="bilinear"
    )


    # -----------------------------
    # RESULT
    # -----------------------------
    # print("Final transform:", utm_ds.GetGeoTransform())
    # print("Final projection:", utm_ds.GetProjection())

    save_tiles(
        utm_ds,
        out_dir=out_folder_path,
        custom_tile_name=custom_tile_name,
        tile_size=out_tile_size,
        overlap=overlap,
        nodata_value=in_nodata_value
    )



def save_tiles(
    utm_ds,
    out_dir:str,
    custom_tile_name:str,
    tile_size:int=512,
    overlap:int=256,
    nodata_value:int=0
):
    """utm_ds is given as a utm-projected gdal raster dataset.
    if no custom tile_name is provided, tiles are named tiled_xxxxx based on number"""
    os.makedirs(out_dir, exist_ok=True)

    arr = utm_ds.ReadAsArray() # reads the utm-proj geotiff as array (h,w) for single band
    gt = utm_ds.GetGeoTransform() # get gdal geotransform
    proj = utm_ds.GetProjection() # and proj

    height, width = arr.shape # get total size of large geotiff

    stride = int(tile_size - overlap) 
    tile_id = 0 # tile counter

    driver = gdal.GetDriverByName("GTiff") # init gdal geotiff driver

    # iterate in x, y coordinates of the image with stride
    for y in range(0, height - tile_size + 1, stride):
        for x in range(0, width - tile_size + 1, stride):

            # extract pixel values.
            tile = arr[y:y + tile_size, x:x + tile_size]

            # skip tile if all nodata
            if (tile == nodata_value).all():
                continue

            # compute geotransform for new tile
            new_gt = (
                gt[0] + x * gt[1] + y * gt[2],
                gt[1],
                gt[2],
                gt[3] + x * gt[4] + y * gt[5],
                gt[4],
                gt[5]
            )
            
            if not custom_tile_name:
                custom_tile_name = 'tile'
            out_path = os.path.join(out_dir, f'{custom_tile_name}_{tile_id:05d}.tif') #:05d adds 0padding e.g. tile 42 --> '00042'

            # create gdal geotiff dataset
            ds_tile = driver.Create(
                out_path,
                tile_size,
                tile_size,
                1,
                utm_ds.GetRasterBand(1).DataType
            )

            ds_tile.SetGeoTransform(new_gt)
            ds_tile.SetProjection(proj)

            band = ds_tile.GetRasterBand(1)
            band.WriteArray(tile) # save out tile
            # band.SetNoDataValue(nodata_value)

            ds_tile.FlushCache()
            ds_tile = None

            tile_id += 1

    print(f'Saved {tile_id} tiles to {out_dir}')

def get_folder_date(in_folder_path):
    folder_name = os.path.basename(in_folder_path)
    if folder_name.startswith('S1'):
        match = re.search(r'(\d{8})T(\d{6})', folder_name)
        if match:
            date = match.group(1)  # Date (YYYYMMDD)
            time = match.group(2)  # Time (hhmmss)

            # Convert to format YYYY_MM_DD_hhmmss
            new_filename = f'{date[:4]}_{date[4:6]}_{date[6:8]}_{time}'
            print(new_filename)
    elif folder_name.startswith('20'):
        new_filename = folder_name
    return new_filename

def get_vv_raster(safe_folder_path):
    vv_raster = None
    for file in os.listdir(f'{safe_folder_path}/measurement'):
        if '-vv-' in file and file.endswith('.tiff'):
            vv_raster = file
            break
    assert vv_raster is not None, f'Error: vv_raster not found in {safe_folder_path}'
    
    vv_raster_path = safe_folder_path + '/measurement/' + vv_raster
    return vv_raster_path