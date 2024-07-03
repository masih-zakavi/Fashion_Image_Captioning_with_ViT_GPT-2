'''
The fashiongen dataset found on Kaggle is in the HDF5 format.
preprocess_h5_file extracts relevant details and builds a csv file
containing the data, and a folder with images
'''

import h5py
import pandas as pd
import numpy as np
from PIL import Image
import os


train_file_path = 'fashiongen_256_256_train.h5'
validation_file_path = 'fashiongen_256_256_validation.h5'

def process_h5_file(file_path, num_of_samples, code, output_image_dir):
    os.makedirs(output_image_dir, exist_ok=True)
    with h5py.File(file_path, 'r') as h5_file:
        images = h5_file['input_image'][:]
        categories = h5_file['input_category'][:]
        descriptions = h5_file['input_description'][:]
        product_IDs = h5_file['input_productID'][:]
        names = h5_file['input_name'][:]
        poses = h5_file['input_pose'][:]
        
        data = []
        for i in range(num_of_samples):
            image_file_name = f'image_{code}_{i}.png'
            
            image = Image.fromarray(np.uint8(images[i]))
            image.save(os.path.join(output_image_dir, image_file_name))
            
            data.append((image_file_name, names[i], categories[i], \
                poses[i], product_IDs[i], descriptions[i]))
        
        df = pd.DataFrame(data, columns=['image_file_name', 'name', 
            'category', 'pose', 'product_ID', 'description'])
        
        df.to_csv(f'{code}_data.csv', index=False)

        print(f"Completed Processing all {code} files")
    
    return data

train_image_dir = 'fashiongen_train_images'
val_image_dir = 'fashiongen_val_images'
test_image_dir = 'fashiongen_test_images'

train_data = process_h5_file(train_file_path, \
            8000, "train" , train_image_dir)
validation_data = process_h5_file(validation_file_path, \
            1000, "val", val_image_dir)
test_data = process_h5_file(validation_file_path, \
            1000, "test" , test_image_dir)

all_data = train_data + validation_data + test_data

df = pd.DataFrame(all_data, columns=['image_file_name', 'name', 
    'category', 'pose', 'product_ID', 'description'])
df.to_csv('fashiongen_data.csv', index=False)