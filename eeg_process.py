import torch
import numpy as np
import os

#Load .pth file
eeg_data = torch.load('./datasets/eeg_5_95_std.pth')

tensor_item = eeg_data['dataset'][0]

# Define a base path to save
base_save_path = "./DreamDiffusion-main/datasets/mne_data"

# Loop over all attributes in the tensor_item
for key, value in tensor_item.items():

    # Construct the subfolder path based on the attribute
    subfolder_path = os.path.join(base_save_path, key)

    # Check if the subfolder exists, if not, create it
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)

    # Save the value to a numpy file
    try:
        if isinstance(value, torch.Tensor):
            ndarray = value.numpy()
        else:
            ndarray = np.array(value)
        np.save(f"{subfolder_path}/0.npy", ndarray)
        print(f"Saved file '{key}.npy' successfully.")
    except Exception as e:
        print(f"Error saving file '{key}.npy': {e}")

#---------------------------------------------------------------------

# import numpy as np
#
# # 指定 .npy 文件路径
# file_path = './dreamdiffusion/datasets/mne_data/eeg/0.npy'
#
# # 使用 NumPy 的 load 函数读取 .npy 文件
# data = np.load(file_path)
#
# # 打印读取的数据
# print(data)

#----------------------------------------------------------------------------
# import numpy as np
# data = np.load('./dreamdiffusion/datasets/mne_data/eeg/0.npy')
#
# # Print its attributes
# print("Shape:", data.shape)
# print("Data Type:", data.dtype)
# print("First few entries:", data)

#---------------------------------------------------------------------------------
dataset = eeg_data['dataset']

# Define a base path to save
base_save_path = "./dreamdiffusion/datasets/mne_data"

# Loop through all items in the dataset
for idx, tensor_item in enumerate(dataset):

    # Loop over all attributes in the tensor_item
    for key, value in tensor_item.items():

        # Construct the subfolder path based on the attribute
        subfolder_path = os.path.join(base_save_path, key)

        # Check if the subfolder exists, if not, create it
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)

        # If the value is a torch.Tensor, convert it to a numpy array
        if isinstance(value, torch.Tensor):
            ndarray = value.numpy()
            try:
              np.save(f"{subfolder_path}/{idx}.npy", ndarray)
            except Exception as e:
              print(f"Error saving file at index {idx}: {e}")
        #else:
            # If the value is not a tensor, simply save it as it is
            #try:
            #  np.save(f"{subfolder_path}/{idx}.npy", np.array(value))    openai/clip-vit-large-patch14
            #except Exception as e:
            #  print(f"Error saving file at index {idx}: {e}")

# Path to the directory
dir_path = './dreamdiffusion/datasets/mne_data/eeg'

# List all files in the directory
files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]

# Print the number of files
print(f"There are {len(files)} files in the directory.")