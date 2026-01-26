# This module contains the functions used for the dataset generation needed as input and target for the deep-learning model

import os
from pathlib import Path
import torch 
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import re

from osgeo import gdal 
from torch.utils.data import TensorDataset 

from preprocessing.satellite_analysis_pre import count_pixels
from preprocessing.satellite_analysis_pre import load_avg

def load_image_array(path, scaled_classes=True):
    '''
    Load a single image using Gdal library. Convert and return it into a numpy array with dtype = np.float32.
    It is implemented and tested to work with JRC collection exported in grayscale (i.e., one channel with pixel values 0, 1, and 2).

    It can also scale the original pixel values by setting the new classes as follows:
            - no-data: -1
            - non-water: 0
            - water: 1
    This is needed for the following binarisation and algorithm implementation (Negstive=0, Positive=1).

    Inputs: 
           path = str, contains full path of the image to be shown
           scaled_classes = bool, sets whether pixel classes are scaled to the range [-1, 1] or kept within the original one [0, 2]
                            default: True, pixel classes are scaled (recommended). 
    
    Output: 
           img_array = np.array, 2D array representing the loaded image
    '''
    # load image 
    img = gdal.Open(path)
    # read it as array
    img_array = img.ReadAsArray().astype(np.float32)

    # scale the pixel value for each class with the updated classification
    if scaled_classes:

       img_array = img_array.astype(int)
       img_array[img_array==0] = -1
       img_array[img_array==1] = 0
       img_array[img_array==2] = 1
    
    return img_array

def create_dir_list(train_val_test, dir_folders=r'data\satellite\dataset', collection=r'JRC_GSW1_4_MonthlyHistory'):
    ''' 
    Get list of path of training, validation and testing datasets.

    Inputs:
           train_val_test = str, specifies what the images are used for.
                            available options: 'training', 'validation' and 'testing'
           dir_folders = str, directory where folders are stored
                         default: r'data\satellite\dataset'
           collection = str, specifies the satellite images collection.
                        default: r'JRC_GSW1_4_MonthlyHistory', the function is implemented to work only with this dataset
    
    Output:
           list_dir = list, contains paths of training, validation and testing dataset folders
    ''' 
    list_dir = []

    # get list of all folders
    for item in os.listdir(dir_folders):
        # get all existing directory
        if os.path.isdir(os.path.join(dir_folders, item)):
            # get only directories that match collection and usage 
            if (train_val_test in item) & (collection in item):
                list_dir.append(os.path.join(dir_folders, item))
    
    # sort by reach id
    list_dir.sort(key=lambda x: int(x.split(f'_{train_val_test}_r')[-1]))
    return list_dir

def create_list_images(train_val_test, reach, dir_folders=r'data\satellite\dataset', collection=r'JRC_GSW1_4_MonthlyHistory'):
    '''
    Return the paths of the satellite images present within a folder given use and reach. 
    It will be used later for loading and creating the dataset.

    Inputs:
           train_val_test = str, specifies what the images are used for.
                            available options: 'training', 'validation' and 'testing'
           reach = int, representing reach number. Number increases going upstream
                   For training, the available range is 1-28 (included). 
                   For validation and testing there is only 1 reach
           dir_folders = str, directory where folders are stored
                         default: r'data\satellite\dataset'
           collection = str, specifies the satellite images collection.
                        default: r'JRC_GSW1_4_MonthlyHistory', the function is implemented to work only with this dataset
    
    Outputs:
            list_dir_images = list, contains the path for each image of the dataset needed for loading it                     
    '''
    # create folder path
    folder = os.path.join(str(dir_folders), collection + rf'_{train_val_test}_r{reach}')
    list_dir_images = []
    # loop through images of that folder
    for image in os.listdir(folder):
        # get only .tif files
        if image.endswith('.tif'):
            path_image = os.path.join(folder, image)
            #print(path_image)
            list_dir_images.append(path_image)
    return list_dir_images




#NEW FUNCTION 

def extract_year_and_reach(path: str):
    fname = Path(path).name  # e.g. "1988_03_01_training_r1.tif"

    # Year: first 4 digits at start
    m_year = re.match(r"^(19|20)\d{2}", fname)
    if not m_year:
        raise ValueError(f"No valid year found in filename: {fname}")
    year = int(m_year.group())

    # Reach: r + digits anywhere
    m_reach = re.search(r"r\d+", fname)
    if not m_reach:
        raise ValueError(f"No reach (r#) found in filename: {fname}")
    reach = m_reach.group()
    return year, reach




def create_datasets(train_val_test, reach, year_target=5, nodata_value=-1, dir_folders=r'data\satellite\dataset', 
                    collection=r'JRC_GSW1_4_MonthlyHistory', scaled_classes=True):
    '''
    Create the input and target dataset for each specific use and reach. Return two lists of lists. 
    The input list has n-elements, with n depending on the year of prediction: 
    if fifth year is predicted the list has four elements. 
    The target list has one element (the year of prediction).

    n-to-1 predictions

    Generate binary images by replaciong `no-data` pixels with the average images stored in the .csv files available in 'data\satellite\averages'. 
    The average images are computed with the function 'get_good_avg' stored in the module 'satellite_analysis_pre'.  
    
    Inputs: 
           train_val_test = str, specifies what the images are used for.
                            available options: 'training', 'validation' and 'testing'
           reach = int, representing reach number. Number increases going upstream.
                   For training, the available range is 1-28 (included)
                   For validation and testing there is only 1 reach
           year_target = int, sets the year predicted after a sequence of input years.
                         default: 5, input dataset is made of 4 images and 5th year is the predicted one
           nodata_value = int, represents pixel value of no-data class.
                          default: -1, based on the updated pixel classes. 
                          If `scaled_classes` = False, this should be set to 0
           dir_folders = str, directory where folders are stored
                         default: r'data\satellite\dataset'
           collection = str, specifies the satellite images collection.
                        default: r'JRC_GSW1_4_MonthlyHistory', the function is implemented to work only with this dataset
           scaled_classes = bool, sets whether pixel classes are scaled to the range [-1, 1] or kept within the original one [0, 2]
                            default: True, pixel classes are scaled (recommended). 
    
    Outputs:
            input_dataset, target_dataset = lists of lists, contain the input and target images respectively  
    '''
    # create list of images (paths)
    list_dir_images = create_list_images(train_val_test, reach, dir_folders, collection)
    # load list of images (arrays)
    images_array = [load_image_array(list_dir_images[i], scaled_classes=scaled_classes) for i in range(len(list_dir_images))]
    # load season averages
    avg_imgs = [load_avg(train_val_test, reach, year, dir_averages=r'data\satellite\averages') for year in range(1988, 1988 + len(images_array))]
    # replace missing data - images are now binary!
    good_images_array = [np.where(image==nodata_value, avg_imgs[i], image) for i, image in enumerate(images_array)]
        
    #print(f'list_dir_images: {list_dir_images}')

    input_dataset = []
    target_dataset = []
    years = []
    reach = []
    
    #print(list_dir_images)
    #print(len(list_dir_images))

    #print(list_dir_images[0])
    extract_year_and_reach(list_dir_images[0])

    '''
    for i in range(len(good_images_array) - year_target + 1):
        path = good_images_array[i]              # e.g. your tif path
        print(path)
        fname = Path(path).name                  # "1988_03_01_training_r17.tif"
        print(fname)
        year = int(fname[:4])                    # 1988
        print(year)
        years.append(year)
    '''
    # loop through images to append these in the originally empty lists
    
    for i in range(len(good_images_array)-year_target+1): # add +1 at the end to include last available year
        input_dataset.append(good_images_array[i:i+year_target-1])
        #print(i)
        target_dataset.append([good_images_array[i+year_target-1]])
        years.append(extract_year_and_reach(list_dir_images[i])[0])
        reach.append(extract_year_and_reach(list_dir_images[i])[1])
    #print(input_dataset, target_dataset, years, reach)
    return input_dataset, target_dataset, years, reach

def combine_datasets(train_val_test, reach, year_target=5, nonwater_threshold=480000, nodata_value=-1, nonwater_value=0,   
                     dir_folders=r'data\satellite\dataset', collection=r'JRC_GSW1_4_MonthlyHistory', scaled_classes=True):
    '''
    Filter the images based on `non-water` amount of pixels threshold. If the requirement is not met, the full inputs-target combination is discarded.
    Select the best images for training the model. After the averaging step, `no-data` pixels are replaced with the season and neighboours average 
    (see 'get_good_avg' function in the module 'satellite_analysis_pre').
    If the full image is composed of `no-data`, the resulting average wil be a fully `non-water` image. These images are discarded.

    Inputs:
           train_val_test = str, specifies what the images are used for.
                            available options: 'training', 'validation' and 'testing'
           reach = int, representing reach number. Number increases going upstream.
                   For training, the available range is 1-28 (included)
                   For validation and testing there is only 1 reach
           year_target = int, sets the year predicted after a sequence of input years.
                         default: 5, input dataset is made of 4 images and 5th year is the predicted one
           nonwater_threshold = int, min amount of `non-water` pixels allowed in the inputs-target combinations
                                default: 480000, necessary to filter out only the fully `non-water` images
           nodata_value = int, represents pixel value of no data class.
                          default: -1, based on the updated pixel classes. 
                          If `scaled_classes` = False, this should be set to 0
           nonwater_value = int, represents pixel value of non-water class.
                         default: 0, based on the updated pixel classes. 
                          If `scaled_classes` = False, this should be set to 1
           dir_folders = str, directory where folders are stored
                         default: r'data\satellite\dataset'
           collection = str, specifies the satellite images collection.
                        default: r'JRC_GSW1_4_MonthlyHistory', the function is implemented to work only with this dataset
           scaled_classes = bool, sets whether pixel classes are scaled to the range [-1, 1] or kept within the original one [0, 2]
                            default: True, pixel classes are scaled (recommended).
                        
    Output:
           filtered_input_dataset, filtered_target_dataset = lists, contain adequate image combinations for input and target datasets, respectively,
                                                             based on `non-water` threshold
    '''
    input_dataset, target_dataset, years, reach_ids = create_datasets(
        train_val_test, reach, year_target, nodata_value, dir_folders, collection, scaled_classes
    )

    filtered_input_dataset, filtered_target_dataset = [], []
    filtered_years, filtered_reaches = [], []
    # filter pairs based on the specified threshold
    for input_images, target_image, year, reach_id in zip(input_dataset, target_dataset, years, reach_ids):
        input_combs = []
        # check input images
        for img in input_images:
            nonwater_count = count_pixels(img, nonwater_value) < nonwater_threshold 
            input_combs.append(nonwater_count)

        # check if input images are all suitable  
        if all(input_combs):
            # check target images
            target_nonwater_thr = count_pixels(target_image[0], nonwater_value) < nonwater_threshold
            if target_nonwater_thr:
                # convert input images to tensor
                input_tensor = [img for img in input_images]
                # convert target image to tensor
                target_tensor = target_image[0]
                
                filtered_input_dataset.append(input_tensor)
                filtered_target_dataset.append(target_tensor)
                filtered_years.append(year)
                filtered_reaches.append(reach_id)
    return filtered_input_dataset, filtered_target_dataset, filtered_years, filtered_reaches








def load_ci_norm_vector(
    reach: str,
    year: int,
    months=(5, 6, 7, 8, 9, 10),
    csv_path=r"preprocessing/Brahmaputra_merged_output.csv",
    fill_value=np.nan,
) -> np.ndarray:
    """
    Load CI_normalized values for a specific reach + year and return a vector for selected months.

    Parameters
    ----------
    reach : str
        Reach identifier exactly as in the CSV, e.g. 'r1', 'r2', ...
    year : int
        Year to select, e.g. 2005
    months : tuple[int]
        Months to include, e.g. (5,6,7,8,9,10)
    csv_path : str
        Path to the CI CSV
    fill_value : float or str
        Value used when a month is missing. Use "reach_mean" to fill with the
        mean CI_normalized for that reach (default NaN).

    Returns
    -------
    np.ndarray
        Vector of shape (len(months),) with CI_normalized values in the same order as `months`.
    """
    df = pd.read_csv(csv_path)

    # Validate expected columns
    required_cols = {"r", "month", "CI_normalized"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}. Available: {list(df.columns)}")

    # Parse month column to datetime
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    if df["month"].isna().any():
        bad = df[df["month"].isna()].head(5)
        raise ValueError(f"Unparseable values in 'month'. Example rows:\n{bad}")

    # Filter to reach + year
    sub = df[(df["r"] == reach) & (df["month"].dt.year == year)].copy()
    reach_df = df[df["r"] == reach]
    if isinstance(fill_value, str) and fill_value == "reach_mean":
        if reach_df.empty:
            raise ValueError(f"No rows found for reach='{reach}' in CSV.")
        reach_series = reach_df["CI_normalized"].dropna()
        if reach_series.empty:
            raise ValueError(f"All CI_normalized values are NaN for reach='{reach}'.")
        reach_mean = float(reach_series.mean())
        fill_value_resolved = reach_mean
    else:
        fill_value_resolved = fill_value
    if sub.empty:
        return np.full(len(months), fill_value_resolved, dtype=np.float32)

    # Extract month number
    sub["m"] = sub["month"].dt.month

    # If multiple rows exist for the same month, average them
    monthly = sub.groupby("m")["CI_normalized"].mean()

    # Build vector in requested month order
    vec = np.array([monthly.get(m, fill_value_resolved) for m in months], dtype=np.float32)
    if np.isnan(vec).any():
        vec = np.where(np.isnan(vec), fill_value_resolved, vec).astype(np.float32)
    return vec


"""
def create_full_dataset(train_val_test, year_target=5, nonwater_threshold=480000, nodata_value=-1, nonwater_value=0, dir_folders=r'data\satellite\dataset', 
                        collection=r'JRC_GSW1_4_MonthlyHistory', scaled_classes=True, device='cuda:0', dtype=torch.int64, ):
    '''
    Generate the full dataset for the given use, combining all reaches.
    Stack all different pairs within one use in order to have the dataset ready for the training, validation and testing of the model.

    Inputs:
           train_val_test = str, specifies what the images are used for.
                            available options: 'training', 'validation' and 'testing'
           year_target = int, sets the year predicted after a sequence of input years.
                         default: 5, input dataset is made of 4 images and 5th year is the predicted one
           nonwater_threshold = int, min amount of `non-water` pixels allowed in the inputs-target combinations
                                default: 480000, necessary to filter out only the fully `non-water` images 
           nodata_value = int, represents pixel value of no data class.
                          default: -1, based on the updated pixel classes. 
                          If `scaled_classes` = False, this should be set to 0
           nonwater_value = int, represents pixel value of non-water class.
                            default: 0, based on the updated pixel classes. 
                            If `scaled_classes` = False, this should be set to 1
           dir_folders = str, directory where folders are stored
                         default: r'data\satellite\dataset'
           collection = str, specifies the satellite images collection.
                        default: r'JRC_GSW1_4_MonthlyHistory', the function is implemented to work only with this dataset
           scaled_classes = bool, sets whether pixel classes are scaled to the range [-1, 1] or kept within the original one [0, 2]
                            default: True, pixel classes are scaled (recommended).
           device = str, specifies device where memory is allocated for performing the computations
                    default: 'cuda: 0' (GPU), other availble option: 'cpu'
           dtype = class, specifies the data type for torch.tensor method.
                   default: torch.int64, it also accepts `torch.float32` to allow gradient computation and backpropagation
    
    Output:
           dataset = TensorDataset, contains all coupled input-target samples for each reach and use
    '''
    # initialize stacked dictionaries
    stacked_dict = {'input': [], 'target': []}
    for folder in os.listdir(dir_folders):
        if train_val_test in folder:
            # get all available reaches
            reach_id = folder.split('_r',1)[1]
            inputs, target = combine_datasets(train_val_test, int(reach_id), year_target, nonwater_threshold, 
                                              nodata_value, nonwater_value, dir_folders, collection, scaled_classes)
            stacked_dict['input'].extend(inputs)
            stacked_dict['target'].extend(target)
       
    # create tensors
    if dtype == None:
        input_tensor = torch.tensor(stacked_dict['input'], device=device)
        target_tensor = torch.tensor(stacked_dict['target'], device=device)
    else:
        input_tensor = torch.tensor(stacked_dict['input'], dtype=dtype, device=device)
        target_tensor = torch.tensor(stacked_dict['target'], dtype=dtype, device=device)
    
    dataset = TensorDataset(input_tensor, target_tensor)
    return dataset
"""
# ----------------------------------------- # 
# TEMPORAL SPLIT #
# ----------------------------------------- # 

def split_list(train_val_test, reach, month, year_end_train=2009, year_end_val=2015, dir_folders=r'data\satellite', collection=r'JRC_GSW1_4_MonthlyHistory'):
    '''
    Split the list created for the dataset generation into three sub-lists, training, validation and testing, respectively.
    Split the lists by years, given the `year_end_train` and `year_end_val` values.
    
    This approach is used for generating a temporal dataset.
    
    Inputs:
           train_val_test = str, specifies what the images are used for.
                            available options: 'training', 'validation' and 'testing'
           reach = int, representing reach number. Number increases going upstream
                   For training, the available range is 1-28 (included)
                   For validation and testing there is only 1 reach
           month = int, represents month of the year from which images are taken. 
                   Available options: 1, 2, 3 or 4 (non-monsoon season months with low-flow conditions). 
                   If another value is given an Exception is raised.
           year_end_train = int, sets last image year used for the training
                            default: 2009, recommended to have 21 years of training data
           year_end_val = int, sets last image year used for the validation
                          default: 2015, recommended to have 5 years of both validation and testing data
           dir_folders = str, directory where datasets are stored
                         default: r'data\satellite'
           collection = str, specifies the satellite images collection.
                        default: r'JRC_GSW1_4_MonthlyHistory', the function is implemented to work only with this dataset
    
    Output:
           train_list, val_list, test_list = lists, contain the path for each training, validation and testing image of the dataset needed for loading it  
    '''
    if month not in {1, 2, 3, 4}:
        raise Exception(f'The specified month is {month}, which is not allowed. It can be either 1, 2, 3 or 4.')
    
    # get monthly dataset directory
    dir_dataset = os.path.join(dir_folders, fr'dataset_month{month}')    
    list = create_list_images(train_val_test, reach, dir_folders=dir_dataset, collection=collection)
    
    # initialize lists
    train_list, val_list, test_list = [], [], [] 
    for path in list:
        # get year of the image
        year = int((path.split('\\')[-1]).split('_')[0])
        # training list
        if year <= year_end_train:
            train_list.append(path)
        # validation list
        elif year_end_train < year <= year_end_val:
            val_list.append(path)
        # testing list
        elif year > year_end_val:
            test_list.append(path)
    return train_list, val_list, test_list










def parse_year_from_path(path: str) -> int:
    m = re.search(r"(19|20)\d{2}", str(path))
    if not m:
        raise ValueError(f"Could not parse a 4-digit year from path: {path}")
    return int(m.group(0))

def reach_to_str(reach_id: int) -> str:
    # adjust if your CSV uses a different reach format
    return f"r{int(reach_id):02d}"

def normalize_reach_id(reach_id: str, train_val_test: str) -> str:
    """
    Normalize a reach id to the CI CSV format.
    - testing  -> r00
    - validation -> r01
    - training -> r1..r28 (no zero padding)
    """
    m = re.search(r"\d+", str(reach_id))
    if not m:
        raise ValueError(f"Could not parse reach id from: {reach_id}")
    reach_num = int(m.group())
    if train_val_test == "testing":
        return "r00"
    if train_val_test == "validation":
        return "r01"
    if train_val_test == "training":
        return f"r{reach_num}"
    raise ValueError(f"Unknown train_val_test: {train_val_test}")

def parse_reach_int(reach_id: str) -> int:
    """
    Parse a reach id like 'r17' or '17' into an integer.
    """
    m = re.search(r"\d+", str(reach_id))
    if not m:
        raise ValueError(f"Could not parse reach id from: {reach_id}")
    return int(m.group())

def build_ci_tensors(years, reaches, train_val_test, year_target=5, months=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
                     csv_path=r"preprocessing/Brahmaputra_merged_output.csv", fill_value=np.nan):
    """
    Build CI tensors aligned with the input/target samples created by create_datasets.

    years and reaches should be the lists returned by create_datasets (years = start year per sample).
    Returns:
        ci_inputs: (N, T, M) where T=year_target-1 and M=len(months)
    """
    if len(years) != len(reaches):
        raise ValueError(f"Years and reaches must match length. Got {len(years)} and {len(reaches)}.")
    T = year_target - 1
    ci_inputs = []
    for start_year, reach_id in zip(years, reaches):
        reach_str = normalize_reach_id(reach_id, train_val_test)
        start_year = int(start_year)
        ci_T = []
        for t in range(T):
            ci_vec = load_ci_norm_vector(
                reach=reach_str,
                year=start_year + t,
                months=months,
                csv_path=csv_path,
                fill_value=fill_value,
            )
            ci_T.append(ci_vec)
        ci_inputs.append(np.stack(ci_T, axis=0))
    ci_inputs = torch.tensor(np.stack(ci_inputs, axis=0), dtype=torch.float32)
    return ci_inputs


def create_full_dataset(train_val_test, year_target=5, nonwater_threshold=480000, nodata_value=-1, nonwater_value=0, dir_folders=r'data\satellite\dataset', 
                        collection=r'JRC_GSW1_4_MonthlyHistory', scaled_classes=True, device='cuda:0', dtype=torch.int64, ci_csv_path=r"preprocessing/Brahmaputra_merged_output.csv", ci_months=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), ci_fill_value="reach_mean"):
    '''
    Generate the full dataset for the given use, combining all reaches.
    Stack all different pairs within one use in order to have the dataset ready for the training, validation and testing of the model.

    Inputs:
           train_val_test = str, specifies what the images are used for.
                            available options: 'training', 'validation' and 'testing'
           year_target = int, sets the year predicted after a sequence of input years.
                         default: 5, input dataset is made of 4 images and 5th year is the predicted one
           nonwater_threshold = int, min amount of `non-water` pixels allowed in the inputs-target combinations
                                default: 480000, necessary to filter out only the fully `non-water` images 
           nodata_value = int, represents pixel value of no data class.
                          default: -1, based on the updated pixel classes. 
                          If `scaled_classes` = False, this should be set to 0
           nonwater_value = int, represents pixel value of non-water class.
                            default: 0, based on the updated pixel classes. 
                            If `scaled_classes` = False, this should be set to 1
           dir_folders = str, directory where folders are stored
                         default: r'data\satellite\dataset'
           collection = str, specifies the satellite images collection.
                        default: r'JRC_GSW1_4_MonthlyHistory', the function is implemented to work only with this dataset
           scaled_classes = bool, sets whether pixel classes are scaled to the range [-1, 1] or kept within the original one [0, 2]
                            default: True, pixel classes are scaled (recommended).
           device = str, specifies device where memory is allocated for performing the computations
                    default: 'cuda: 0' (GPU), other availble option: 'cpu'
           dtype = class, specifies the data type for torch.tensor method.
                   default: torch.int64, it also accepts `torch.float32` to allow gradient computation and backpropagation
    
    Output:
           dataset = TensorDataset, contains inputs, targets, and CI inputs per sample
    '''
    # initialize stacked dictionaries
    stacked_dict = {'input': [], 'target': []}
    stacked_years = []
    stacked_reaches = []
    for folder in os.listdir(dir_folders):
        #print(folder)
        if train_val_test in folder:
            # get all available reaches
            reach_id = folder.split('_r',1)[1]
            inputs, target, years, reaches = combine_datasets(
                train_val_test,
                int(reach_id),
                year_target,
                nonwater_threshold,
                nodata_value,
                nonwater_value,
                dir_folders,
                collection,
                scaled_classes,
            )
            
            stacked_dict['input'].extend(inputs)
            stacked_dict['target'].extend(target)
            stacked_years.extend(years)
            stacked_reaches.extend(reaches)
        
    # create tensors
    if dtype is None:
        input_tensor = torch.tensor(stacked_dict['input'])        # removed device=device
        target_tensor = torch.tensor(stacked_dict['target'])      # removed device=device
    else:
        input_tensor = torch.tensor(stacked_dict['input'], dtype=dtype)       # removed device=device
        target_tensor = torch.tensor(stacked_dict['target'], dtype=dtype)     # removed device=device
    ci_tensor = build_ci_tensors(
        stacked_years,
        stacked_reaches,
        train_val_test=train_val_test,
        year_target=year_target,
        months=ci_months,
        csv_path=ci_csv_path,
        fill_value=ci_fill_value,
    )
    years_tensor = torch.tensor(stacked_years, dtype=torch.int64)
    reaches_tensor = torch.tensor([parse_reach_int(r) for r in stacked_reaches], dtype=torch.int64)
    dataset = TensorDataset(input_tensor, target_tensor, ci_tensor, years_tensor, reaches_tensor)
    
    return dataset

def create_split_datasets(train_val_test, reach, month, use_dataset, year_end_train=2009, year_end_val=2015, year_target=5, nodata_value=-1,
                          dir_folders=r'data\satellite', collection=r'JRC_GSW1_4_MonthlyHistory', scaled_classes=True):
    '''
    Creates the input and target datasets for each specific use and reach. Return two lists of lists.
    The input list has n-elements, with n depending on the year of prediction: 
    if fifth year is predicted the list has four elements. 
    The target list has one element (the year of prediction).

    n-to-1 predictions

    Generate binary images by replaciong `no-data` pixels with the average images stored in the .csv files available in 'data\satellite\averages'. 
    The average images are computed with the function 'get_good_avg' stored in the module 'satellite_analysis_pre'.
    
    Inputs: 
           train_val_test = str, specifies what the images are used for.
                            available options: 'training', 'validation' and 'testing'
           reach = int, representing reach number. Number increases going upstream.
                   For training, the available range is 1-28 (included)
                   For validation and testing there is only 1 reach
           month = int, represents month of the year from which images are taken. 
                   Available options: 1, 2, 3 or 4 (non-monsoon season months with low-flow conditions). 
           use_dataset = str, specifies what the generated dataset is used for.
                         available options: ' training', ' validation', 'testing'
           year_end_train = int, sets last images year used for the training
                            default: 2009, recommended to have 21 years of training data
           year_end_val = int, sets last images year used for the validation
                          default: 2015, recommended to have 5 years of both validation and testing data
           year_target = int, sets the year predicted after a sequence of input years.
                         default: 5, input dataset is made of 4 images and 5th year is the predicted one
           nodata_value = int, represents pixel value of no data class.
                          default: -1, based on the updated pixel classes. 
                          If `scaled_classes` = False, this should be set to 0
           dir_folders = str, directory where folders are stored
                         default: r'data\satellite\dataset'
           collection = str, specifies the satellite images collection.
                        default: r'JRC_GSW1_4_MonthlyHistory', the function is implemented to work only with this dataset
           scaled_classes = bool, sets whether pixel classes are scaled to the range [-1, 1] or kept within the original one [0, 2]
                            default: True, pixel classes are scaled (recommended). 
    
    Outputs:
            input_dataset, target_dataset = lists of lists, contain the input and target images respectively  
    '''
    # list of images' paths
    train_list, val_list, test_list = split_list(train_val_test, reach, month, year_end_train, year_end_val, dir_folders, collection)
    
    if use_dataset == 'training':
        # list of images (arrays)
        images_train = [load_image_array(train_list[i], scaled_classes=scaled_classes) for i in range(len(train_list))]
        # load average images
        avg_train = [load_avg(train_val_test, reach, year, dir_averages=r'data\satellite\averages') for year in range(1988, 1988 + len(images_train))]
        # replace missing data
        good_images_train = [np.where(image==nodata_value, avg_train[i], image) for i, image in enumerate(images_train)]
        # initialize lists
        input_train, target_train = [], []
        for i in range(len(good_images_train)-year_target):
            input_train.append(good_images_train[i:i+year_target-1])
            target_train.append([good_images_train[i+year_target-1]])
        train = [input_train, target_train] 
        return train
    
    # similar procedure repeated for validation and testing datasets
    elif use_dataset == 'validation':
        images_val = [load_image_array(val_list[i], scaled_classes=scaled_classes) for i in range(len(val_list))]
        avg_val = [load_avg(train_val_test, reach, year, dir_averages=r'data\satellite\averages') for year in range(year_end_train, year_end_train + len(images_val))]
        good_images_val = [np.where(image==nodata_value, avg_val[i], image) for i, image in enumerate(images_val)]
        input_val, target_val = [], [] 
        for i in range(len(good_images_val)-year_target):
            input_val.append(good_images_val[i:i+year_target-1])
            target_val.append([good_images_val[i+year_target-1]])
        val = [input_val, target_val] 
        return val

    elif use_dataset == 'testing':
        images_test = [load_image_array(test_list[i], scaled_classes=scaled_classes) for i in range(len(test_list))]
        avg_test = [load_avg(train_val_test, reach, year, dir_averages=r'data\satellite\averages') for year in range(year_end_val, year_end_val + len(images_test))]
        good_images_test = [np.where(image==nodata_value, avg_test[i], image) for i, image in enumerate(images_test)]
        input_test, target_test = [], []
        for i in range(len(good_images_test)-year_target):
            input_test.append(good_images_test[i:i+year_target-1])
            target_test.append([good_images_test[i+year_target-1]])
        test = [input_test, target_test]
        return test
    
    else:
        raise Exception(f'The given use_dataset is {use_dataset} but is wrong.\n\
The possible choices are `training`, `validation`, `testing`.')

def combine_split_datasets(train_val_test, reach, month, use_dataset, year_end_train=2009, year_end_val=2015, year_target=5, nonwater_threshold=480000, 
                           nodata_value=-1, nonwater_value=0, dir_folders=r'data\satellite', collection=r'JRC_GSW1_4_MonthlyHistory', scaled_classes=True):
    '''
    Filter the images based on `non-water` amount of pixels threshold. If the requirement is not met, the full inputs-target combination is discarded.
    Select the best images for training the model. After the averaging step, `no-data` pixels are replaced with the season and neighboours average 
    (see 'get_good_avg' function in the module 'satellite_analysis_pre').
    If the full image is composed of `no-data`, the resulting average wil be a fully `non-water` image. These images are discarded.

    Inputs:
           train_val_test = str, specifies what the images are used for.
                            available options: 'training', 'validation' and 'testing'
           reach = int, representing reach number. Number increases going upstream.
                   For training, the available range is 1-28 (included)
                   For validation and testing there is only 1 reach
           month = int, represents month of the year from which images are taken. 
                   Available options: 1, 2, 3 or 4 (non-monsoon season months with low-flow conditions). 
           use_dataset = str, specifies what the generated dataset is used for.
                         available options: ' training', ' validation', 'testing'
           year_end_train = int, sets last images year used for the training
                            default: 2009, recommended to have 21 years of training data
           year_end_val = int, sets last images year used for the validation
                          default: 2015, recommended to have 5 years of both validation and testing data
           year_target = int, sets the year predicted after a sequence of input years.
                         default: 5, input dataset is made of 4 images and 5th year is the predicted one
           nonwater_threshold = int, max amount of no-data pixels allowed in the images
                                default: 480000, necessary to filter out only the fully `non-water` images
           nodata_value = int, represents pixel value of no data class.
                          default: -1, based on the updated pixel classes. 
                          If `scaled_classes` = False, this should be set to 0
           nonwater_value = int, represents pixel value of non-water class.
                         default: 0, based on the updated pixel classes. 
                          If `scaled_classes` = False, this should be set to 1
           dir_folders = str, directory where folders are stored
                         default: r'data\satellite\dataset'
           collection = str, specifies the satellite images collection.
                        default: r'JRC_GSW1_4_MonthlyHistory', the function is implemented to work only with this dataset
           scaled_classes = bool, sets whether pixel classes are scaled to the range [-1, 1] or kept within the original one [0, 2]
                            default: True, pixel classes are scaled (recommended).
                        
    Output:
           filtered_input_dataset, filtered_target_dataset = lists, contain adequate images combinations for input and target dataset, respectively,
                                                             based on non-water threshold
    '''
    dataset = create_split_datasets(train_val_test, reach, month, use_dataset, year_end_train, year_end_val, 
                                             year_target, nodata_value, dir_folders, collection, scaled_classes)
    
    # get inputs and targets
    input_dataset, target_dataset = dataset[0], dataset[1]

    filtered_inputs, filtered_targets = [], []

    # filtering pairs based on the specified threshold
    for input_images, target_image in zip(input_dataset, target_dataset):
        input_combs = []
        # check input images
        for img in input_images:
            nonwater_count = count_pixels(img, nonwater_value) < nonwater_threshold 
            input_combs.append(nonwater_count)
        # check if input images are all suitable
        if all(input_combs):
            # check target images
            target_nonwater_thr = count_pixels(target_image[0], nonwater_value) < nonwater_threshold
            if target_nonwater_thr:
                # convert input images to tensor
                input_tensor = [img for img in input_images]
                # convert target image to tensor
                target_tensor = target_image[0]
                
                filtered_inputs.append(input_tensor)
                filtered_targets.append(target_tensor)
      
    filtered_dataset = [filtered_inputs, filtered_targets]
    return filtered_dataset

def create_split_dataset(month, use_dataset, year_target=5, year_end_train=2009, year_end_val=2015, nonwater_threshold=480000, nodata_value=-1, nonwater_value=0, 
                         dir_folders=r'data\satellite', collection=r'JRC_GSW1_4_MonthlyHistory', scaled_classes=True, device='cuda:0', dtype=torch.int64):
    '''
    Generate the full dataset for the given use, combining all reaches.
    Stack all different pairs within one use in order to have the dataset ready for the training, validation and testing of the model.

    Inputs:
           month = int, represents month of the year from which images are taken. 
                   Available options: 1, 2, 3 or 4 (non-monsoon season months with low-flow conditions). 
           use_dataset = str, specifies what the generated dataset is used for.
                         available options: ' training', ' validation', 'testing'
           year_target = int, sets the year predicted after a sequence of input years.
                         default: 5, input dataset is made of 4 images and 5th year is the predicted one
           year_end_train = int, sets last images year used for the training
                            default: 2009, recommended to have 21 years of training data
           year_end_val = int, sets last images year used for the validation
                          default: 2015, recommended to have 5 years of both validation and testing data
           nonwater_threshold = int, max amount of no-data pixels allowed in the images
                                default: 480000, necessary to filter out only the fully `non-water` images
           nodata_value = int, represents pixel value of no data class.
                          default: -1, based on the updated pixel classes. 
                          If `scaled_classes` = False, this should be set to 0
           nonwater_value = int, represents pixel value of non-water class.
                         default: 0, based on the updated pixel classes. 
                          If `scaled_classes` = False, this should be set to 1
           dir_folders = str, directory where folders are stored
                         default: r'data\satellite\dataset'
           collection = str, specifies the satellite images collection.
                        default: r'JRC_GSW1_4_MonthlyHistory', the function is implemented to work only with this dataset
           scaled_classes = bool, sets whether pixel classes are scaled to the range [-1, 1] or kept within the original one [0, 2]
                            default: True, pixel classes are scaled (recommended).
           device = str, specifies device where memory is allocated for performing the computations
                    default: 'cuda:0' (GPU), other availble option: 'cpu'
           dtype = class, specifies the data type for torch.tensor method.
                   default: torch.int64, it also accepts `torch.float32` to allow gradient computation and backpropagation
    
    Output:
           dataset = TensorDataset, contains all coupled input-target samples for each reach and use
    '''
    train_val_test = ['training', 'validation', 'testing']
    dir_dataset = os.path.join(dir_folders, rf'dataset_month{month}')
    stacked_dataset= {'input': [], 'target': []}
    
    # loop over folder and use 
    for folder in os.listdir(dir_dataset):
        for use in train_val_test:
            if use in folder:
                # loop through all reaches
                reach_id = folder.split('_r',1)[1]
                filtered_dataset = combine_split_datasets(
                    use, reach_id, month, use_dataset, year_end_train, year_end_val, 
                    year_target, nonwater_threshold, nodata_value, nonwater_value,  
                    dir_folders, collection, scaled_classes
                    )
    
                stacked_dataset['input'].extend(filtered_dataset[0])
                stacked_dataset['target'].extend(filtered_dataset[1])
       
    inputs, targets = torch.tensor(stacked_dataset['input'], dtype=dtype, device=device), torch.tensor(stacked_dataset['target'], dtype=dtype, device=device)
    dataset = TensorDataset(inputs, targets)
    return dataset
