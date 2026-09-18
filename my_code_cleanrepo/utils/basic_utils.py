import os
import re
import zipfile
from glob import glob

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


def save_gdf_as_zipped_shapefile(gdf, folder, zip_name, remove_shp_after_zip:bool):
    os.makedirs(folder, exist_ok=True)
    shp_path = os.path.join(folder, 'custom.shp')
    gdf.to_file(shp_path, driver='ESRI Shapefile')
    
    file_list = glob(os.path.join(folder, '*'))
    file_list = [x for x in file_list if not x.endswith('zip')]

    zip_path = os.path.join(folder, zip_name)
    zip_files(file_list, zip_path)

    if remove_shp_after_zip:
        for file in file_list:
            os.remove(file)

def zip_files(file_list, zip_path):
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file in file_list:
            zipf.write(file, arcname=os.path.basename(file))


def parallel_unzip(in_folder_path:str):
    import zipfile
    from pathlib import Path
    import concurrent.futures
    from datetime import datetime
    import time

    counter = {
        'i': 0,
        'total': 0
    }

    def extract_here(zip_path):
        # safe_folder_path = zip_path.replace('.zip','.SAFE')

        zip_path = Path(zip_path)
        extract_to = zip_path.parent

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            print(f"Extracted : {zip_path.name}")
        except Exception as e:
            print(f"Failed to extract {zip_path.name}: {e}")

        counter['i'] += 1
        now = datetime.now()
        print(f"{now} — {counter['i']}/{counter['total']}")

    def unzip_files(folder_path, max_workers=20
                                , stagger_delay=0.5
                                ):
        folder = Path(folder_path)
        zip_files = [zip_file for zip_file in folder.glob("*.zip") if not (zip_file.with_suffix('.SAFE')).exists()]
        to_remove = [zip_file for zip_file in folder.glob("*.zip") if (zip_file.with_suffix('.SAFE')).exists()]
        counter['total'] = len(zip_files)

        for f in to_remove:
            os.remove(f)
            print(f'Deleted {f}')

        if not zip_files:
            print("No zip files found.")
            return

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for zip_file in zip_files:
                futures.append(executor.submit(extract_here, zip_file))
                time.sleep(stagger_delay)  # Stagger the thread starts to avoid burst I/O

    # 
    unzip_files(
        in_folder_path, 
        max_workers=20, 
        stagger_delay=0.4
        )