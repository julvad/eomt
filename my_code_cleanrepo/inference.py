import os 
import numpy as np
import torch
import segmentation_models_pytorch as smp
from typing import Union, Sequence, Literal, Optional, List
import geopandas as gpd
from shapely.geometry import shape
from smoothify import smoothify
from shapely import Polygon, MultiPolygon
from PIL import Image
import rasterio
from rasterio.merge import merge
from rasterio.features import shapes
from torchmetrics.classification import MulticlassF1Score, MulticlassJaccardIndex
import json
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    raise AssertionError("CUDA not available")
from my_code.utils.transforms import sar_transform
from my_code.segformer_galeio.segformer import SegmentationModelSegformer
import random
from rasterio.enums import Resampling

def predict_pytorch(
        model:torch.nn.Module, 
        tile_paths:Union[str, Sequence[str]],
        transform,
        predict_prob_or_class:Literal['probs','class']='class',
        return_dice_iou:bool=False,
        save_out_preds:bool=False,
        out_pred_path:str='preds', 
        resize_mask_px:Optional[int] = None,
        batch_size:int=8,
        shuffle:bool=True):
    """
    model: loaded pytorch nn module,
    tile_paths: path to tile or tiles folder,
    transform: torchvision transform as used in training,
    return_dice_or_iou: if true, there need to be corresponding 'images' and 'labels'
    out_pred_path: folder path
    """
    assert not (return_dice_iou and predict_prob_or_class=='probs'),'return dice/iou needs class preds, not probs'
    if save_out_preds:
        os.makedirs(out_pred_path, exist_ok=True)

    if not isinstance(tile_paths, list):
        if os.path.isfile(tile_paths) and tile_paths.endswith('.tif'): # If tile_paths is single image
            tile_paths = [tile_paths]
        else:
            raise AttributeError(tile_paths)

    ll = len(tile_paths)
    if shuffle:
        random.shuffle(tile_paths)

    if return_dice_iou:
        iou = MulticlassJaccardIndex(num_classes=2, average="macro").to(device)
        dice = MulticlassF1Score(num_classes=2, average="macro").to(device)

    model.eval()
    model.to(device)
    for start in range(0, ll, batch_size):
        end = min(start + batch_size, ll) # end idx
        batch_paths = tile_paths[start:end]  # batch_imgs patchs

        # Build the sub-batch_imgs of tensors
        batch_imgs = []
        batch_masks = []

        for img_path in batch_paths:
            # with rasterio.open(img_path) as src:
            #     data = src.read() # N BANDS
            #     img_tensor = transform(data) # as tensor
            with Image.open(img_path) as img_pil: #NOTE: sometimes PIL bugs with uint16 multiband rasters?
                img_tensor = transform(img_pil)
            batch_imgs.append(img_tensor)

            # append masks
            if return_dice_iou:
                mask_path = img_path.replace('images','labels')
                assert os.path.isfile(mask_path)
                with rasterio.open(mask_path) as src:
                     # Resample data while reading
                    if resize_mask_px:
                        mask = src.read(
                            out_shape=(src.count, resize_mask_px, resize_mask_px),
                            resampling=Resampling.nearest
                        ).squeeze()
                    else:
                        mask = src.read().squeeze() # label band
                    batch_masks.append(torch.from_numpy(mask).short())

        batch_torch = torch.stack(batch_imgs, dim=0).to(device)

        # predict 
        with torch.no_grad():
            outputs = model(batch_torch)               # logits (B, 2, H, W)
            if predict_prob_or_class == 'probs':
                # probability maps for class 1
                pred_masks = outputs.softmax(dim=1) # (B, 2, H, W) softmax probs
                pred_masks = pred_masks[:, 1]  ## GET PROBS FOR CLASS 1 (0 = bg) # (B,H,W)
            elif predict_prob_or_class == 'class':
                # class map
                pred_masks = outputs.argmax(dim=1) # argmax for classification
            else:
                raise AttributeError(predict_prob_or_class)

        # pred_masks = pred_masks.detach().cpu().numpy()
        # print(f'Pred_masks shape: {pred_masks.shape}')
        if return_dice_iou:
             # batch_masks_np = np.stack(batch_masks, axis=0)
            batch_masks_torch = torch.stack(batch_masks, dim=0).to(device)  
            
            iou.update(pred_masks, batch_masks_torch)
            dice.update(pred_masks, batch_masks_torch)
            # iou.update(pred_masks, batch_masks_np)
            # dice.update(pred_masks, batch_masks_np)

            print(
                f"Batch {start}/{ll}:"
                f"Dice={dice.compute().item()}, "
                f"IoU={iou.compute().item()}"
            )

        ## SAVE PRED MASKS IN OUT PRED PATH
        if save_out_preds:
            for pred_mask, img_path in zip(pred_masks, batch_paths):
                img_name = os.path.basename(img_path)
                out_path = os.path.join(out_pred_path, img_name)

                with rasterio.open(img_path) as src:
                    profile = src.profile.copy()
                
                if predict_prob_or_class=='class':
                    profile.update( # Keep the same profile as the in img, but 1 band.
                        count=1,
                        dtype=rasterio.uint8
                    )
                    pred_mask = pred_mask.astype(np.int8)

                elif predict_prob_or_class=='probs':
                    profile.update(
                        count=1,
                        dtype=rasterio.float32 # allow to save as float
                    )
                    pred_mask = pred_mask.astype(np.float32)
                else:
                    raise AttributeError

                with rasterio.open(out_path, 'w', **profile) as dst:
                    dst.write(pred_mask, 1)

        # after each batch_imgs: clear memory
        del batch_torch, outputs, pred_masks 
        # c+=batch_size
    
        # print(f'Pred {c}/{ll}: Saved predictions for {img_name}') 
    if return_dice_iou:
        return round(dice.compute().item(), 3), round(iou.compute().item(), 3)
    torch.cuda.empty_cache() 
    

def predict_multiscale_ensemble(
        ts_ps_tuples:List[tuple]=[(384,16),(512,20),(448,33)],
        tiles_path_dict:dict={512:['path.tif','path2.tif']},
        model_path_dict:dict = {
            512:'models/512-20/UNetPlusPlus_efficientnet-b5_unfrz_bs10_lr1e-4_ce-loss/best_model_epoch26_iou0.793.pth',
            384:'models/384-16/UNetPlusPlus_efficientnet-b3_unfrz_bs24_lr1e-4_ce-loss/best_model_epoch104_iou0.797.pth',
            448:'models/448-33/UNetPlusPlus_efficientnet-b4_unfrz_bs16_lr1e-4_ce-loss/best_model_epoch22_iou0.802.pth'
            },
        transform=None,
        predict_prob_or_class:Literal['probs','class']='probs',
        out_preds_folder_path:str = r'c:\Users\juvad3723\.1\--Data\cruise_2026',
        save_out_preds=True,
        batch_size=8
    ):
    """
    dict with keys=tile sizes.
    """

    for ts,ps in ts_ps_tuples:
        if 'SegformerGaleio' in model_path_dict[ts]:
            mit_size = model_path_dict[ts].split('SegformerGaleio')[1][5:7]#get mit size in string
            model = SegmentationModelSegformer( # load model from galeio segformer
                mit_size=mit_size,
                num_classes=2,
                size_output=(ts, ts),
                device='cuda:0',
                head_fuse_dropout=0.1,
            )
            state_dict = torch.load(model_path_dict[ts]) # load model weights
            model.load_state_dict(state_dict['model_state_dict'])
            triple = True # this model uses tripled in bands
        else:
            model = smp.from_pretrained(model_path_dict[ts]) # load model from smp (with weights)
            # my smp models are unfortunately inconsistent with the number of in_channels
            with open(os.path.join(model_path_dict[ts], 'config.json'), 'r') as j:
                model_config = json.load(j)  # loads smp model config, a JSON file into a Python dict
                triple = True if model_config['in_channels']==3 else False # 
                
        if not transform:
            transform = sar_transform(resize_size=ts,triple=triple) # define transform

        out_pred_path = os.path.join(out_preds_folder_path, f'{ts}_{ps}')
        tiles_path_list = tiles_path_dict[ts]
        #Predict
        predict_pytorch(
            model=model,
            tile_paths=tiles_path_list,
            transform=transform,
            predict_prob_or_class=predict_prob_or_class,
            return_dice_iou=False,
            save_out_preds=save_out_preds,
            out_pred_path=out_pred_path,
            batch_size=batch_size
        )
        del model
        torch.cuda.empty_cache()
    

def mosaic_with_rasterio(
        in_tile_paths, 
        out_mosaic_path, 
        method:Literal['mean','min','max','sum']='mean',
        padding:Union[int, Literal['auto','no_padding']]='auto', # int: padding with a number | auto: 1/4 tile padding | no_padding
        out_mosaic_dtype:Literal['float','int']='float',
        mosaic_classify_softmax:bool=False,
        softmax_threshold:float=0.5
        ):
    """
    Assumes single-band rasters.
    """
    os.makedirs(os.path.dirname(out_mosaic_path), exist_ok=True)
    if mosaic_classify_softmax:
        assert out_mosaic_dtype=='int', 'if classify output, out dtype must be int'

    tiles_rio = [rasterio.open(f) for f in in_tile_paths] # open all tiles as a rasterio dataset
    padded_datasets = []
    memory_files = []

    nodata_val = 255 #nodata for uint8

    if padding == 'no_padding':
        padded_datasets = tiles_rio # no padding
    else: # TODO: Fix padding! Right now is giving more false positives...
        if isinstance(padding,int): # Pad with a fixed pad size 
            pad_px_ds = padding
        for ds in tiles_rio:
            if padding == 'auto': # Or pad adaptively (ex. 1/4 of tile)
                pad_px_ds = ds.width // 8 #TODO: experiment with other paddings?

            # Pad before merging (ignore a number of pixels near tile boundaries)    
            padded_data = pad_dataset_inward(ds, pad_px_ds, padding_value=nodata_val)

            # Write padded data to an in-memory dataset for merging ## very fast
            ## get metadata          
            meta = ds.meta.copy()
            meta.update(nodata=nodata_val) # add the nodata val to tile metadata

            mem_ds = rasterio.io.MemoryFile()
            with mem_ds.open(**meta) as m: # Pad with the nodata value
                m.write(padded_data)

            memory_files.append(mem_ds)
            padded_datasets.append(mem_ds.open()) #effective transfer and delete in memory data
            del mem_ds, padded_data

    # Merge by mean
    if method=='mean': 
        # calculate merged sum and count ## TODO: is there a more efficient way of calculating the mean?
        sum_array, out_transform = merge(
            sources=padded_datasets, 
            dtype=np.float32, 
            method='sum',
            nodata=nodata_val,
            masked=True)
        count_array, out_transform = merge(
            sources=padded_datasets, 
            dtype=np.float32, 
            method='count',
            nodata=nodata_val,
            masked=True)

        # Hide RuntimeWarning: invalid value encountered in divide
        with np.errstate(divide='ignore', invalid='ignore'): 
            # calculate mean TODO: is there a more efficient way of calculating the mean?6
            mosaic = sum_array/count_array 

        # Get rid of NaN caused by divide by 0
        mosaic[count_array == 0] = nodata_val
    else:# 'min or max or sum' 
        mosaic, out_transform = merge(
            sources=padded_datasets,
            method=method,
            nodata=nodata_val  # tells merge which values to mask if needed
        )
    with rasterio.open(in_tile_paths[0]) as src: # copy the geospatial metadata from a (the first) tile ## and edit later
        profile = src.profile

    if out_mosaic_dtype == 'int':
        dtype = 'uint8'
    elif out_mosaic_dtype == 'float':
        dtype = 'float32'
    else:
        raise AttributeError(f'wrong dtype: {dtype}')
    
    # Write output
    profile.update( # define output metadata
        height=mosaic.shape[1],
        width=mosaic.shape[2],
        transform=out_transform,
        nodata=nodata_val,
        dtype=dtype,
        count=1,
        compress='lzw',
    )

    if mosaic_classify_softmax: #TODO: Experiment with different thresholds? e.g. >0.5 or >=0.5?
        assert 0<softmax_threshold<1, f'softmax threshold must be a prob between 0 and 1, got {softmax_threshold}'
        # classified = classified != nodata_val
        # classified = mosaic > softmax_threshold
        classified = np.where( # more elegant than above? Make sure that nodata_val doesnt get classified?
            mosaic == nodata_val,
            nodata_val,
            (mosaic > softmax_threshold).astype(np.uint8)
        )
        with rasterio.open(out_mosaic_path, "w", **profile) as dst:
            dst.write(classified.astype(np.uint8))
        del classified
    else:
        with rasterio.open(out_mosaic_path, "w", **profile) as dst:
            if out_mosaic_dtype=='float':
                dst.write(mosaic.astype(np.float32))
            else:
                dst.write(mosaic.astype(np.uint8))
        del mosaic

    for ds in tiles_rio:
        ds.close()
    for mem_ds in padded_datasets:
        mem_ds.close()
    for mf in memory_files:    
        mf.close()
        
    del tiles_rio, padded_datasets, memory_files

def pad_dataset_inward(dataset, padding: int, padding_value=0):
    data = dataset.read()
    data[:, :padding, :]  = padding_value  # top
    data[:, -padding:, :] = padding_value  # bottom
    data[:, :, :padding]  = padding_value  # left
    data[:, :, -padding:] = padding_value  # right
    return data


def mosaic_rasterio_padding( #TODO: add soft mosaicking for probs predictions
    in_tile_paths: list[str], 
    out_mosaic_path: str, 
    pad_px: Union[int, str] = 'auto',
    method:Literal['mean','min','max','sum']='max',
    ):
    """
    Mosaic pred tiles with padding
    """
    out_file_dir = os.path.dirname(out_mosaic_path)
    os.makedirs(out_file_dir,exist_ok=True)
    datasets = [rasterio.open(f) for f in in_tile_paths]

    # Apply inward padding to each tile before merging
    padded_datasets = []
    for ds in datasets:
        if pad_px == 'auto':
            tile_width = ds.block_shapes[0][1]
            pad_px_ds = tile_width // 8
        else:
            pad_px_ds = pad_px
        padded_data = pad_dataset_inward(ds, pad_px_ds)
        # Write padded data to an in-memory dataset for merging
        mem_ds = rasterio.io.MemoryFile()
        with mem_ds.open(**ds.meta) as m:
            m.write(padded_data)
        padded_datasets.append(mem_ds.open())
        del mem_ds

    mosaic, transform = merge(padded_datasets, method=method)

    meta = datasets[0].meta.copy()

    if method == 'mean':
        out_dtype = 'float32'
    else:
        out_dtype = 'uint8'

    meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": transform,
        "count": mosaic.shape[0],
        # "dtype": mosaic.dtype,
        "dtype": out_dtype,
        "compress": "lzw"
    })

    with rasterio.open(out_mosaic_path, "w", **meta) as dest:
        dest.write(mosaic)

    for ds in datasets :
        ds.close()
    for mem_ds in padded_datasets:
        mem_ds.close()
    del datasets, padded_datasets


def raster_to_polygon(
    in_raster_path,
    out_folder,
    min_area_m2=1e4,
    max_area_m2=1e8,
    out_format:Literal['shp','geojson']='shp',
    class_values:list=[1],
    postprocessing:bool=True
    ):
    """Converts a binary geotiff to a polygon shapefile 
    """
    os.makedirs(out_folder,exist_ok=True)
    datename = os.path.basename(in_raster_path).strip('.tif')
    if isinstance(class_values,int):
        class_values = [class_values]

    with rasterio.open(in_raster_path) as src:
        band = src.read(1)
        transform = src.transform

        if len(class_values)>1:
            unique_values = np.unique(band[band != src.nodata]) # get all unique values (classes)
            unique_values = [v for v in unique_values if v in class_values]
            unique_values = unique_values.tolist()
        
        else:
            unique_values = class_values
    
    for unique_value in unique_values:
        # mask for cell values
        mask = band == unique_value # bool array of the same shape
        if not mask.any():
            return # no pixels found (no detections) --> skip
        
        # shapes generates polygons from the mask
        geom_rasterio = shapes(
            band,
            mask=mask,  # only keep the masked cells
            transform=transform  # keep same spatial reference
        )

        # # converts the rasterio geom to shapely geom
        shapely_geom = (
            {"geometry": shape(geom), "properties": {"value": v}}
            for geom, v in geom_rasterio if v == unique_value
        )
            
        # build gdf
        gdf = gpd.GeoDataFrame.from_features(shapely_geom, crs=src.crs)

        # ensure polygon shape # NOTE: NEeded?
        def ensure_polygon(geom):
            if isinstance(geom, Polygon):
                return Polygon(geom.exterior)
            elif isinstance(geom, MultiPolygon):
                # Convert MultiPolygon to single polygon (take convex hull or first polygon)
                return geom.convex_hull  # or geom.buffer(0) for cleanup
            else:
                return geom

        gdf["geometry"] = gdf.geometry.apply(ensure_polygon)

        if postprocessing:
        # detections postprocessing
            gdf = postprocess_gdf(
                gdf,
                min_area_m2=min_area_m2,
                max_area_m2=max_area_m2,
                merge_nearby_dist_m=100,
                smoothen=False, # done in arcpy geospatial analysis
                simplify_geom=False, # done in arcpy geospatial analysis
            )

        # gdf.geometry = gdf.geometry.make_valid() # in all cases ensure valid geometries #NOTE: Experimental line, see if OK, or needed

        if len(unique_values) > 1:
            out_name = f'{datename}_class{unique_value}.{out_format}'
        else:
            out_name = datename + '.' + out_format

        out_path = os.path.join(out_folder,out_name)

        gdf['date'] = datename


        if len(gdf) == 0:
            print(f'Skipping vectorization of {out_path}. No detections.')
            continue

        if out_format == 'shp':
            gdf.to_file(out_path, driver='ESRI Shapefile')
        elif out_format == 'geojson':
            gdf.to_file(out_path, driver='GeoJSON')
        else:
            raise ValueError(out_format)
        
        # print(f'{out_format} created at {out_path}')


def postprocess_gdf(
        gdf,
        merge_nearby_dist_m:Optional[int]=None,
        min_area_m2:Optional[int]=None,
        max_area_m2:Optional[int]=None,
        # remove_holes:bool=False, #TODO: implement; but needed?
        simplify_geom:bool=False,
        smoothen:bool=False,
):
    
    # fix geometries
    # gdf.geometry = gdf.geometry.make_valid()

    # merge nearby polygons
    if merge_nearby_dist_m:
        gdf = merge_nearby(gdf, dist_m=merge_nearby_dist_m)

    # remove small/large polygons
    if min_area_m2:
        gdf = gdf[gdf.geometry.area >= min_area_m2]
    if max_area_m2:
        gdf = gdf[gdf.geometry.area <= max_area_m2]

    # simplify
    if simplify_geom:
        gdf["geometry"] = gdf.geometry.simplify(0.5, preserve_topology=True)

    if smoothen:
        # Apply smoothing (segment_length auto-detected from geometry)
        gdf = smoothify(
            geom=gdf,
            smooth_iterations=1, # More iterations = smoother result
            merge_collection = True  # Merges overlapping polygons
            # num_cores=4  # Use parallel processing for large datasets
        )    

    # if multipolygons:
        # gdf = gdf.dissolve(by="value") #NOTE: broken anyway... but keep this in mind.

    return gdf

# def merge_nearby(gdf, distance_m):
#     buffered = gdf.geometry.buffer(distance_m)           # buffer by 100 m
#     merged = buffered.unary_union                  # merge overlapping buffers
#     merged_polygons = gpd.GeoSeries(merged).buffer(-distance_m)  # remove buffer to original size
#     return gpd.GeoDataFrame(geometry=merged_polygons, crs=gdf.crs)

def merge_nearby(gdf, dist_m=50):
    geom = gdf.geometry.buffer(dist_m).unary_union
    geom = gpd.GeoSeries([geom], crs=gdf.crs).buffer(-dist_m)
    return gpd.GeoDataFrame(geometry=geom, crs=gdf.crs).explode(ignore_index=True)


