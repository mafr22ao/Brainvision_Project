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


