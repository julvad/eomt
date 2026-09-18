import os
import numpy as np
import arcpy
from arcpy.ia import *
import subprocess
from osgeo import gdal
from pyproj import Transformer
import re
from typing import Literal, Optional
from pyproj import CRS
import rasterio
from datetime import datetime

# arcpy.env.workspace = 'memory'

def find_utm_details(in_raster):
    # Get center longitude and latitude in WGS84
    raster = arcpy.Raster(in_raster)
    ext = raster.extent

    center = arcpy.PointGeometry(
        arcpy.Point(
            (ext.XMin + ext.XMax) / 2,
            (ext.YMin + ext.YMax) / 2
        ),
        raster.spatialReference
    ).projectAs(arcpy.SpatialReference(4326))

    lon = center.firstPoint.X
    lat = center.firstPoint.Y

    # UTM zone (1-60)
    utm_zone = int((lon + 180) / 6) + 1

    # UTM EPSG
    if lat >= 0:
        epsg = 32600 + utm_zone   # WGS84 / UTM Northern Hemisphere
    else:
        epsg = 32700 + utm_zone   # WGS84 / UTM Southern Hemisphere

    cm = utm_zone * 6 - 183

    reproj_details = f'PROJCS["WGS_1984_UTM_Zone_{utm_zone}",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],' \
        'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],PARAMETER["False_Easting",500000.0],' \
        f'PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",{cm}.0],PARAMETER["Scale_Factor",0.9996],PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0]]'
    
    return utm_zone, reproj_details, epsg


def find_utm_details_old(in_raster):
    """
    returns arcpy crs details for the corresponding UTM zone of the raster
    return utm_zone, reproj_details, epsg
    """
    # desc = arcpy.Raster(in_raster)
    
    # x_max = desc.extent.XMax
    # x_min = desc.extent.XMin
    desc = arcpy.Raster(in_raster)
    sr = desc.spatialReference

    # Only project if needed
    if sr.factoryCode != 4326:
        ll = arcpy.PointGeometry(desc.extent.lowerLeft, sr).projectAs(arcpy.SpatialReference(4326))
        ur = arcpy.PointGeometry(desc.extent.upperRight, sr).projectAs(arcpy.SpatialReference(4326))

        x_min = ll.firstPoint.X
        x_max = ur.firstPoint.X
    else:
        x_max = desc.extent.XMax
        x_min = desc.extent.XMin

    raster_longitude_center = (x_max + x_min) / 2

    if -126 < raster_longitude_center <= -120:
        utm_zone = '10N'
        cm = -123
    elif -120 < raster_longitude_center <= -114:
        utm_zone = '11N'
        cm = -117
    elif -96 < raster_longitude_center <= -90:
        utm_zone = '15N'
        cm = -93
    elif -90 < raster_longitude_center <= -84:
        utm_zone = '16N'
        cm = -87
    elif -84 < raster_longitude_center <= -78:
        utm_zone = '17N'
        cm = -81
    elif -78 < raster_longitude_center <= -72:
        utm_zone = '18N'
        cm = -75
    elif -72 < raster_longitude_center <= -66:
        utm_zone = '19N'
        cm = -69
    elif -66 < raster_longitude_center <= -60:
        utm_zone = '20N'
        cm = -63 #central meridian 
    elif -60 < raster_longitude_center <= -54:
        utm_zone = '21N'
        cm = -57 #central meridian 
    elif -54 < raster_longitude_center <= -48:
        utm_zone = '22N'
        cm = -51 #central meridian 
    elif -48 < raster_longitude_center <= -42:
        utm_zone = '23N'
        cm = -45 #central meridian
    elif -42 < raster_longitude_center <= -36:
        utm_zone = '24N'
        cm = -39 #central meridian
    elif -6 < raster_longitude_center <= 0:
        utm_zone = '30N'
        cm = -3 #central meridian 
    elif 0 < raster_longitude_center <= 6:
        utm_zone = '31N'
        cm = 3 #central meridian 
    elif 6 < raster_longitude_center <= 12:
        utm_zone = '32N'
        cm = 9 #central meridian 
    elif 12 < raster_longitude_center <= 18:
        utm_zone = '33N'
        cm = 15
    elif 18 < raster_longitude_center <= 24:
        utm_zone = '34N'
        cm = 21
    elif 24 < raster_longitude_center <= 30:
        utm_zone = '35N'
        cm = 27
    elif 30 < raster_longitude_center <= 36:
        utm_zone = '36N'
        cm = 33
    elif 36 < raster_longitude_center <= 42:
        utm_zone = '37N'
        cm = 39
    elif 42 < raster_longitude_center <= 46:
        utm_zone = '38N'
        cm = 45
    elif 46 < raster_longitude_center <= 54:
        utm_zone = '39N'
        cm = 51
    elif 132 < raster_longitude_center <= 138:
        utm_zone = '53N'
        cm = 135
    elif 138 < raster_longitude_center <= 144:
        utm_zone = '54N'
        cm = 141
    elif 144 < raster_longitude_center <= 150:
        utm_zone = '55N'
        cm = 147
    elif 150 < raster_longitude_center <= 156:
        utm_zone = '56N'
        cm = 153
    elif 156 < raster_longitude_center <= 162:
        utm_zone = '57N'
        cm = 159
    elif 162 < raster_longitude_center <= 168:
        utm_zone = '58N'
        cm = 165
    else:
        raise ValueError(f"Error: UTM zone not in the list. Longitude center of input raster: {raster_longitude_center}.")
    # print(f'UTM zone: {utm_zone}')
    
    reproj_details = f'PROJCS["WGS_1984_UTM_Zone_{utm_zone}",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],' \
        'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],PARAMETER["False_Easting",500000.0],' \
        f'PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",{cm}.0],PARAMETER["Scale_Factor",0.9996],PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0]]'
    z = utm_zone.strip('N')
    epsg = int(f'326{z}')

    return utm_zone, reproj_details, epsg

def find_s1_sat(safe_folder_path):
    s1 = None
    safe_folder = os.path.basename(safe_folder_path)
    # Find which S1 satellite
    if safe_folder.startswith('S1') and safe_folder.endswith('.SAFE'):
        s1 = safe_folder[:3]
    elif safe_folder.startswith('20'):
        for file in os.listdir(os.path.join(safe_folder_path,'measurement')):
            if not file.startswith('s1'):
                continue
            s1 = file[:3].upper()
            break
    assert s1 is not None, f'Error: s1 Sat not found for {safe_folder_path}'
    return s1

def land_mask_reproj(in_raster, out_pixel_size=10, land_mask=True, utm_project=True):
    with arcpy.EnvManager(
        overwriteOutput=True,
        pyramid="NONE",
        rasterStatistics="NONE"
    ):
        out_raster = None
        utm_zone, reproj_details, epsg = find_utm_details(in_raster)
        if land_mask: ## find corresponding binary raster (water = 1, land = 0)
            landmask_path = r'c:\Users\juvad3723\.1\--Data\land_mask\world_landmask_2500m.tif' # NOTE: New world landmask. see perfo if change
            clipped = arcpy.ia.Clip(
                in_raster=in_raster,
                aoi=landmask_path
            )
            if utm_project:
                sr = arcpy.SpatialReference(epsg)  
                out_raster = 'memory/projected.crf'
                arcpy.management.Delete(out_raster)
                arcpy.management.ProjectRaster(
                    in_raster=clipped, 
                    out_raster=out_raster, 
                    out_coor_system=sr, 
                    resampling_type='BILINEAR',
                    cell_size=out_pixel_size)
            
                print('done: land mask and reproject')
            else: 
                out_raster = arcpy.ia.Con(
                    in_conditional_raster = landmask_path,
                    in_true_raster_or_constant = in_raster,
                    in_false_raster_or_constant = None,
                    where_clause = ''
                )
                print('done: land mask')
        else:
            if utm_project:
                out_raster = 'memory/reprojected.crf' #NOTE: if this fails, remove .crf or replace with .tif
                arcpy.management.Delete(out_raster)
                arcpy.management.ProjectRaster(
                    in_raster=in_raster,
                    out_raster=out_raster,
                    out_coor_system=reproj_details,
                    resampling_type='BILINEAR',
                    cell_size=out_pixel_size,
                    geographic_transform=None,
                    Registration_Point=None,
                    vertical="NO_VERTICAL"
                )
                print('done: reproject')
            else:
                out_raster = in_raster
                print('Warning, no land-mask or reprojection done.')
        if out_raster is not None:
            
            return out_raster

def land_mask_reproj_old(in_raster, out_pixel_size=10, land_mask=True, utm_project=True): 
    """This old version relies on a hard-coded utm zone finder and landmask finder.
    New version (above) uses a world ocean mask, and is a bit faster with .crf format!"""
    out_raster = None
    utm_zone, reproj_details, epsg = find_utm_details(in_raster)
    if land_mask: ## find corresponding binary raster (water = 1, land = 0)
        land_mask_path = get_landmask_path(in_raster)

        if utm_project:
            with arcpy.EnvManager(outputCoordinateSystem = reproj_details, cellSize = out_pixel_size, resamplingMethod='BILINEAR', pyramid='NONE'):
                out_raster = arcpy.ia.Con(
                    in_conditional_raster = land_mask_path,
                    in_true_raster_or_constant = in_raster,
                    in_false_raster_or_constant = None,
                    where_clause = ''
                )
            print('done: land mask and reproject')
        else: 
            with arcpy.EnvManager(pyramid='NONE'):
                out_raster = arcpy.ia.Con(
                    in_conditional_raster = land_mask_path,
                    in_true_raster_or_constant = in_raster,
                    in_false_raster_or_constant = None,
                    where_clause = ''
                )
            print('done: land mask')
    else:
        if utm_project:
            with arcpy.EnvManager(pyramid='NONE'):
                out_raster = 'memory/reprojected.crf' #NOTE: if this fails, remove .crf or replace with .tif
                arcpy.management.ProjectRaster(
                    in_raster=in_raster,
                    out_raster=out_raster,
                    out_coor_system=reproj_details,
                    resampling_type='BILINEAR',
                    cell_size=out_pixel_size,
                    geographic_transform=None,
                    Registration_Point=None,
                    vertical="NO_VERTICAL"
                )
            print('done: reproject')
            
        else:
            out_raster = in_raster
            print('Warning, no land-mask or reprojection done.')
    if out_raster is not None:
        return out_raster

def get_landmask_path(in_raster):
    """returns a path to a tif landmask""" ##binary raster (water = 1, land = 0)
    
    utm_zone, reproj_details, epsg = find_utm_details(in_raster)
    if utm_zone in ['10N','11N']:
        land_mask_path = r'c:\Users\juvad3723\.1\--Data\land_mask\california_landmask.tif'
    elif utm_zone in ['15N','16N']:
        land_mask_path = r'c:\Users\juvad3723\.1\--Data\land_mask\gulfofmexico_landmask.tif' 
    elif utm_zone in ['20N','21N']:
        land_mask_path = r'c:\Users\juvad3723\.1\--Data\land_mask\stlawrence_landmask.tif'  
    elif utm_zone in ['30N','31N','32N','33N']:
        land_mask_path = r'c:\Users\juvad3723\.1\--Data\land_mask\northsea_barents_landmask.tif'
    elif utm_zone in ['34N','35N','36N','37N','38N']:
        desc = arcpy.Raster(in_raster)
        y_min = desc.extent.YMin
        if epsg != 4326: # if raster is projected (not wgs 84)
            y_min = Transformer.from_crs(epsg, 4326, always_xy=True).transform(0, y_min)[1]
        if y_min < 70:
            land_mask_path = r'c:\Users\juvad3723\.1\--Data\land_mask\blacksea_azov_landmask.tif'
        else:
            land_mask_path = r'c:\Users\juvad3723\.1\--Data\land_mask\barentssea_landmask.tif'
    elif utm_zone == '39N':
        land_mask_path = r'c:\Users\juvad3723\.1\--Data\land_mask\caspian_persic_landmask.tif'
    elif utm_zone in ['53N','54N','55N','56N','57N','58N']:
        land_mask_path = r'c:\Users\juvad3723\.1\--Data\land_mask\okhotsk_landmask.tif'
    
    else:
        raise Exception(f'Error: No existing land mask for UTM zone {utm_zone}')
    
    return land_mask_path

def orbit_correction(GRD_folder_path):
    assert os.path.isdir(GRD_folder_path), 'Error: input should be a GRD SAFE folder path'

    in_manifest_path = f'{GRD_folder_path}/manifest.safe'
    assert os.path.isfile(in_manifest_path), 'Error: manifest.safe file not found'

    # remove existing orbit files
    for file in os.listdir(GRD_folder_path):
        if file.endswith('.EOF'):
            file_path = os.path.join(GRD_folder_path,file)
            os.remove(file_path)

    #___________________________________________________
    # download orbit file
    for ot in ['SENTINEL_PRECISE','SENTINEL_RESTITUTED']:
        try:
            arcpy.ia.DownloadOrbitFile(
                in_radar_data = in_manifest_path,
                orbit_type=ot,
                username='',
                password='',
                cloud_storage=None,
                folder=None
            )
            #if file is downloaded: apply orbit correction
            out_raster = arcpy.ia.ApplyOrbitCorrection(
                in_radar_data = in_manifest_path,
                # in_orbit_file=None,
                # folder=None
            )   
            print(f'Done: orbit correction with {ot}')
            
            break
        except:
            if ot == 'SENTINEL_RESTITUTED':
                # If restituted also isnt available, proceed with uncorrected orbit image.
                out_raster = in_manifest_path
                print(f'Error: No orbit file found. Proceeded without orbit correction')
            continue
    return out_raster

def remove_thermal_noise(in_raster, pol_bands='VV'):
    
    out_raster = arcpy.ia.RemoveThermalNoise(
        in_radar_data=in_raster,
        polarization_bands=pol_bands
    )
    print('Done: thermal noise removal')
    return out_raster

def radiometric_correction(in_raster, calibration_type: Literal['SIGMA_NOUGHT', 'BETA_NOUGHT', 'GAMMA_NOUGHT', 'NONE'], pol_bands='VV'): 
    #___________________________________________________
    # apply radiometric corrections
    # in_raster should be a GRD IW Raster with a single VV band
    assert calibration_type in ['SIGMA_NOUGHT', 'BETA_NOUGHT', 'GAMMA_NOUGHT'], "Error: calibration_type should be one of ['SIGMA_NOUGHT', 'BETA_NOUGHT', 'GAMMA_NOUGHT']"
    if calibration_type != 'NONE':
        try: 
            out_raster = arcpy.ia.ApplyRadiometricCalibration(
                in_radar_data = in_raster,
                polarization_bands = pol_bands,
                calibration_type = calibration_type
            )
            print(f'Done: calibration to {calibration_type}')
        except:
            out_raster = in_raster
            print(f'Error during {calibration_type} calibration: procceeded without radiometric calibration')
        return out_raster
    else:
        return in_raster
    
def terrain_correction(in_raster, pol_bands='VV', dem_path=None):
    out_raster = arcpy.ia.ApplyGeometricTerrainCorrection(
        in_radar_data=in_raster,
        polarization_bands=pol_bands,
        in_dem_raster=dem_path,
        geoid="GEOID"
    )
    print('Done: terrain correction')
    return out_raster
##
def speckle_filter(in_raster):
    filter = r'c:\Users\juvad3723\.1\--Projects\GitHub\north_sea_dl-mapping\raster_functions\_3x3_lee_filter.rft.xml'
    out_raster = arcpy.ia.Apply(
        in_raster = in_raster, 
        raster_function = filter)
    print('Done: fast despeckling')
    return out_raster

##
def linear_to_db(in_raster, calculate_stats=True):
    if calculate_stats:
        stats = 'STATISTICS 1 1'
    else:
        stats = 'NONE'
    try: 
        with arcpy.EnvManager(pyramid="NONE", rasterStatistics=stats): # with stats
            out_raster = arcpy.ia.ConvertSARUnits(
                in_radar_data = in_raster,
                conversion_type = "LINEAR_TO_DB"
            )
        print(f'Done: units converted to dB')
    except:
        out_raster = 10 * arcpy.sa.Log10(in_raster)
        print('Error: arcpy linear to dB failed. Used 10 * Log10 conversion instead')
    return out_raster

##
def calculate_statistics(in_raster, skip_factor=1, ignored_value=None):
    if ignored_value is not None:
        arcpy.management.CalculateStatistics(
            in_raster_dataset=in_raster, 
            x_skip_factor=skip_factor, 
            y_skip_factor=skip_factor, 
            ignore_values=ignored_value, 
            skip_existing=True)
    else:
        arcpy.management.CalculateStatistics(
            in_raster_dataset=in_raster, 
            x_skip_factor=skip_factor, 
            y_skip_factor=skip_factor,
            skip_existing=True)

def normalization_8bit (in_raster, real_stats=True, min_value=Literal[0,1]):
    if real_stats:
        desc = arcpy.Describe(in_raster)
        if not (hasattr(desc, "statistics") and desc.statistics): 
            calculate_statistics(in_raster=in_raster) # If no stats avail, calculate stats

        if min_value==1:
            norm = r'c:\Users\juvad3723\.1\--Projects\GitHub\north_sea_dl-mapping\raster_functions\_2std_stretch_1_255_realStats.rft.xml'
        elif min_value==0:
            norm = r'c:\Users\juvad3723\.1\--Projects\GitHub\north_sea_dl-mapping\raster_functions\_2std_stretch_8bit_realStats.rft.xml'
        else:
            raise BaseException('Error: min_value should be 0 or 1')
    else:
        if min_value==0:
            norm = r'c:\Users\juvad3723\.1\--Projects\GitHub\north_sea_dl-mapping\raster_functions\_2std_stretch_8bit_estStats.rft.xml'
        elif min_value==1:
            norm = r'c:\Users\juvad3723\.1\--Projects\GitHub\north_sea_dl-mapping\raster_functions\_2std_stretch_1_255_estStats.rft.xml'
        else:
            raise BaseException('Error: min_value should be 0 or 1')
        
    norm = os.path.abspath(norm)
    out_raster = arcpy.ia.Apply(
        in_raster = in_raster, 
        raster_function = norm) 
    
    return out_raster

def normalization_float (in_raster, real_stats=True):
    if real_stats:
        desc = arcpy.Describe(in_raster)
        if not (hasattr(desc, "statistics") and desc.statistics): 
            calculate_statistics(in_raster=in_raster) # If no stats avail, calculate stats
        norm = r'c:\Users\juvad3723\.1\--Projects\GitHub\north_sea_dl-mapping\raster_functions\_2std_stretch_float_realStats.rft.xml'
    else:
        norm = r'c:\Users\juvad3723\.1\--Projects\GitHub\north_sea_dl-mapping\raster_functions\_2std_stretch_float_estStats.rft.xml'

    norm = os.path.abspath(norm)
    out_raster = arcpy.ia.Apply(
        in_raster = in_raster, 
        raster_function = norm) 
    
    return out_raster

def standardization_float(in_raster):
    desc = arcpy.Describe(in_raster)
    if not (hasattr(desc, "statistics") and desc.statistics): 
        calculate_statistics(in_raster=in_raster) # If no stats avail, calculate stats
    
def log_transform(in_raster):
    out_raster = arcpy.sa.Log10(in_raster)
    return out_raster

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

def save_raster(in_raster, out_raster_path, 
                pixel_type: Literal['1_BIT','8_BIT_UNSIGNED','16_BIT_UNSIGNED','32_BIT_FLOAT'], 
                no_data_value: Optional[int] = None, 
                calculate_stats=False, 
                calculate_pyramids=False
                ):
    if calculate_stats:
        stats = 'STATISTICS 1 1'
    else:
        stats = 'NONE'
    
    if calculate_pyramids:
        pyramids =  'PYRAMIDS -1 NEAREST DEFAULT 75 NO_SKIP NO_SIPS'
    else:
        pyramids = 'NONE'

    # assert out_raster_path.endswith('.tif'), 'Error: raster output should have .tif extension'

    out_raster_path = os.path.abspath(out_raster_path)
    arcpy.env.overwriteOutput = True
    with arcpy.EnvManager(compression='LZ77', rasterStatistics=stats, pyramid=pyramids):
            arcpy.management.CopyRaster(
                in_raster = in_raster,
                out_rasterdataset = out_raster_path,
                pixel_type=pixel_type,
                format="TIFF",
                config_keyword="",
                background_value=None,
                nodata_value=no_data_value,
                onebit_to_eightbit="NONE",
                colormap_to_RGB="NONE",
                scale_pixel_value="NONE",
                RGB_to_Colormap="NONE",
                transform="NONE",
                process_as_multidimensional="CURRENT_SLICE",
                build_multidimensional_transpose="NO_TRANSPOSE"
            )
    print(f'Raster saved at {out_raster_path}')

    
