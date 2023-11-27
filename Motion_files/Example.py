import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import numpy as np
from mpmath.identification import transforms

# Load optical flow data from the NPZ file
optical_flow_data = np.load("C:/Users/andre/OneDrive/Documents/GitHub/Brainvision_Project/Motion_files/batch_0_Optical_Flow_Stack.npz", allow_pickle=True)
print(optical_flow_data.keys())

# Assuming the optical flow data is stored in a variable named "flow_data"
flow_data = optical_flow_data["Data"]

# Print the shape of the flow_data array to see its dimensions
print("Shape of flow_data:", flow_data.shape)

first_flow_stack = flow_data[0]

# Checking the shape of the first optical flow stack
shape = first_flow_stack.shape
print("Shape of the first optical flow stack:", shape)
# Convert the numpy array to a PyTorch tensor
flow_tensor = torch.from_numpy(flow_data)

# Define a transform to preprocess the optical flow data
transform = transforms.Compose([
    transforms.ToPILImage(),        # Convert to PIL Image
    transforms.Resize((224, 224)),  # Resize to 224x224 (adjust as needed)
    transforms.ToTensor(),          # Convert to tensor
])

# Apply the transform to the optical flow tensor
preprocessed_flow = transform(flow_tensor)

# Define a custom ResNet-101 model with the weights argument
class CustomResNet101(nn.Module):
    def __init__(self, num_classes, in_channels=16):
        super(CustomResNet101, self).__init__()
        # Load the pretrained ResNet-101 model with weights argument
        self.resnet101 = models.resnet101(weights=torchvision.models.resnet.ResNet101_Weights.IMAGENET1K_V1)

        # Modify the first convolutional layer to accept 16 input channels
        self.resnet101.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Modify the output layer to have the desired number of classes
        self.resnet101.fc = nn.Linear(self.resnet101.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet101(x)


# Example usage:
num_classes = 5  # Number of action classes (adjust as needed)
input_channels = 16  # Number of input channels for optical flow data
model = CustomResNet101(num_classes, input_channels)

# You can now use this custom model to train and test on your optical flow data.
# Make sure to prepare your data accordingly and set up the training and evaluation loops.
