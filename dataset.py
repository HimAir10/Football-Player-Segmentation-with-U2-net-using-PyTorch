
import zipfile
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage.io import imread
import numpy as np
import os

class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_file, transform=None):
        self.images_dir = images_dir
        self.masks = np.load(masks_file)  # Assumes masks are stored in a numpy array
        self.image_paths = sorted(os.listdir(images_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.image_paths[idx])
        image = imread(image_path)
        mask = self.masks[idx]

        # Convert image and mask to tensors
        image = torch.tensor(image).permute(2, 0, 1).float() / 255.0  # Normalize image to [0,1]
        mask = torch.tensor(mask).unsqueeze(0).float()  # Add channel dimension for mask

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

# Define transformations (if any)
transform = transforms.Compose([
    transforms.ToTensor(),
    # You can add other transformations like random crop, flip, etc.
])

# Define the zip file and the extraction path
zip_file_path = r"C:\Users\Intern\Downloads\archive (4).zip"
extracted_folder = r"C:\Users\Intern\Downloads\extracted_images"

# Extract the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_folder)

# Print the contents to ensure extraction
("Extracted files:", os.listdir(extracted_folder))

# Now, assuming the 'images' folder is inside the extracted folder
images_dir = os.path.join(extracted_folder, 'images')

# Check if the images folder exists
("Contents of images directory:", os.listdir(images_dir))

# Proceed with the DataLoader using the extracted folder path
train_dataset = SegmentationDataset(images_dir=images_dir, masks_file="masks.npy", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
