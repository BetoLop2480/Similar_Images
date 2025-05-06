# Step 1. Import Libraries

import torch # In this example we utilize Pytorch in the implementation
import torch.nn as nn
# To retrive the pretrained models. It is a subpackage containing defintion of models
# to perform diverse computer vision tasks. 
import torchvision.models as models 
from torchvision import transforms 
from torch.utils.data import DataLoader, Dataset # To handle datasets
from PIL import Image # In this example we do not use OpenCV
import os

import numpy as np
import pandas as pd
import random



# Step 2. Load a Pre-trained Model
# https://docs.pytorch.org/vision/stable/models.html
# Load a pre-trained model in this case we select the RESNET##
#model = models.resnet18(pretrained=True) # --Old fashioned--
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# Freezing all the parameters to prevent gradient computation during feature extraction
# We want to exploit the learned parameters 
for param in model.parameters():
    param.requires_grad = False


# Step 3. Modify the Model for Feature Extraction
# https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html
# Remove the final classification layer
# model.children() -> Retrieve the main layers of the network
# nn.Sequential(*) -> rebuilds the network without the fc layer
feature_extractor = nn.Sequential(*list(model.children())[:-1])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor = feature_extractor.to(device)

# Set the model to evaluation mode
feature_extractor.eval()



# 4. Define Image Transformations
# Define the transformations to apply to the input images
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # Mean and std calculate over the ImageNet dataset using RGB color space.
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])




# 5. Create a Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, img_path




# 6. Extract Features from Images
def extract_features(image_folder, batch_size=32):
    # Get all image paths
    image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) 
                  if img.endswith(('.jpg'))]
    
    # Create dataset and dataloader
    dataset = CustomDataset(image_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Dictionary to store features and corresponding image paths
    features_dict = {}
    
    # Extract features
    with torch.no_grad():
        for batch_images, batch_paths in dataloader:
            batch_images = batch_images.to(device)

            # Forward pass to get features
            batch_features = feature_extractor(batch_images)

            # Remove extra dimensions (from adaptive avg pooling)
            batch_features = batch_features.squeeze()
            
            # Removing extra dimensions to move to the CPU
            batch_features = batch_features.squeeze().cpu()

            # Convert to numpy and store in dictionary
            for features, path in zip(batch_features, batch_paths):
                features_dict[path] = features.numpy()

            # Free GPU memory
            torch.cuda.empty_cache()
                
    return features_dict


def similarity_idx(features_dict, which_image):
	"""
	To determine similarity between images. Top 5 more similar.
	"""

	topN = 5 # Five most similar images
	img_path = list(features_dict.keys())    
    #feat_val = list(features_dict.values())
	distance = pd.DataFrame(columns=['Image', 'Distance'])
	root_FeatVec = features_dict[which_image]

	for img in img_path:
		other_FeatVec = features_dict[img]

		square__dif = (root_FeatVec-other_FeatVec) ** 2
		sum_square_dif = np.sum(square__dif)
		euclidean_distance = np.sqrt(sum_square_dif)
		
		# Nueva fila
		nueva_fila = {'Image': img, 'Distance': euclidean_distance}
		distance.loc[len(distance)] = nueva_fila

	return distance

    

if __name__ == "__main__":

	# Path to your image folder
	image_folder = "C:/Users/alber/Datasets/ECSD/"
    
	# Extract features
	features = extract_features(image_folder)
	np.save("extracted_features.npy", features)
	features_dict = np.load("C:/Users/alber/extracted_features.npy", allow_pickle=True).item()
	
	# Choosing between a particular or random image
	fix_or_rand = True
	key_ = ''
	if fix_or_rand:
		key_ = 'C:/Users/alber/Datasets/ECSD/0004.jpg'
	else:
		key_ = random.choice(list(features_dict.keys()))
	

	print("Finding the similar images to : ", key_)

	df_distance = pd.DataFrame(columns=['Image', 'Distance'])
	df_distance = similarity_idx(features_dict, key_)
	#print(df_distance)

	df_sorted = df_distance.sort_values(by='Distance', ascending=True)
	print(df_sorted.head(3))

	# Checking some dimensions of the extracetd features =)
	#print(f"Extracted features for {len(features_dict)} images")
	#print(f"Feature vector shape: {features_dict[next(iter(features_dict))].shape}")    