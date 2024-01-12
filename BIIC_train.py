import torchvision.models as models
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50
from PIL import Image
import os
import random
from basnet import BASNet
import cv2
from torchvision import transforms


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn


if torch.cuda.is_available():
    print("GPU is available!")
    device = torch.device("cuda")
else:
    print("GPU is not available, using CPU.")
    device = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


class ForegroundDataset(Dataset):
    def __init__(self, root_folder, transform=None):
        self.root_folder = root_folder
        self.transform = transform
        self.classes = os.listdir(root_folder)
        self.image_paths = []

        for class_name in self.classes:
            class_folder = os.path.join(self.root_folder, class_name)
            image_names = os.listdir(class_folder)
            self.image_paths.extend([(class_name, img_name)
                                    for img_name in image_names])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        class_name, img_name = self.image_paths[idx]
        img_path = os.path.join(self.root_folder, class_name, img_name)
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, self.classes.index(class_name)
 # Return image and its class index


class BackgroundDataset(Dataset):
    def __init__(self, root_folder, transform=None):
        self.root_folder = root_folder
        self.transform = transform
        self.img_paths = [os.path.join(root_folder, img_name) for img_name in os.listdir(
            root_folder)]  # List background images

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img


# Define transforms for images (example)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Instantiate dataset objects
foreground_dataset = ForegroundDataset(
    root_folder=r"G:\THESIS\CODE\SaliencyMix-main\SaliencyMix-main\SaliencyMix-ImageNet\dataset\train", transform=transform
)

# Use train folder for foreground images
background_dataset = BackgroundDataset(
    root_folder=r"G:\THESIS\CODE\SaliencyMix-main\SaliencyMix-main\SaliencyMix-ImageNet\dataset\background", transform=transform
)

# Instantiate dataset objects for validation
val_dataset = ForegroundDataset(
    root_folder=r'G:\THESIS\CODE\SaliencyMix-main\SaliencyMix-main\SaliencyMix-ImageNet\dataset\val', transform=transform)

# Synthesized image
save_dir = r'G:\THESIS\CODE\SaliencyMix-main\SaliencyMix-main\SaliencyMix-ImageNet\dataset\result'

# Create data loaders
foreground_loader = DataLoader(
    foreground_dataset, batch_size=4, shuffle=True
)
background_loader = DataLoader(
    background_dataset, batch_size=4, shuffle=True
)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)


# Load BASNet model (adjust paths and loading methods as needed)
basnet = BASNet(3, 1)
basnet.load_state_dict(torch.load(
    r'G:\THESIS\CODE\SaliencyMix-main\SaliencyMix-main\SaliencyMix-ImageNet\basnet\basnet.pth'))

if torch.cuda.is_available():
    basnet.cuda()
    basnet.eval()

# Load ResNet-50 model
resnet_model = models.resnet50(pretrained=True)

# Modify output layer to match number of classes
num_classes = 100  # Replace with your actual number of classes
resnet_model.fc = nn.Linear(2048, num_classes)

# Move model to device
resnet_model = resnet_model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet_model.parameters(), lr=0.001)
num_epochs = 50
image_count = 0
batch_size = 4
image_index = 1  # Initialize a global image index


# Sysnthesized image save in a directory
def save_images(tensor_to_save):
    global image_index  # Access the global image index

    for i in range(tensor_to_save.shape[0]):
        image_tensor = tensor_to_save[i, :, :, :]
        image_tensor = image_tensor.mul(255).clamp(
            0, 255).byte()  # Denormalize (if needed)
        image = Image.fromarray(image_tensor.permute(1, 2, 0).numpy())

        # Construct the full path using the save directory and image index
        save_path = os.path.join(save_dir, f"image_{image_index}.png")

        # Ensure the directory exists before saving
        # Create the directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        image.save(save_path)  # Save the image to the specified path

        image_index += 1  # Increment the index for the next image


for epoch in range(num_epochs):
    print("Training epoch ", epoch)
    iteration = 0

    for foreground_images, foreground_labels in foreground_loader:
        iteration += 1
        print("\tIteration ", iteration)
        # TORCH_USE_CUDA_DSA error solution
        with torch.autograd.detect_anomaly():

            # Change background 40% of the time
            # for i in range(len(foreground_images)):
            if random.random() < 0.5:
                # Your training loop here
                # Get a batch of background images
                background_images = next(iter(background_loader))

                d1, d2, d3, d4, d5, d6, d7, d8 = basnet(
                    foreground_images.to(device))

                # normalization
                pred = d1[:, 0, :, :]
                masks = normPRED(pred)
                # print(masks.shape)
                masks = torch.unsqueeze(masks, 1)
                # print(masks.shape)
                del d1, d2, d3, d4, d5, d6, d7, d8
                masks = masks.detach().cpu()
                result = foreground_images * masks + \
                    background_images * (1 - masks)
                # print(result.shape)

                # synthesize image saving in particular directory
                # for i in range(0, len(result), batch_size):
                #     batch_tensor = result[i:i+batch_size]
                #     save_images(batch_tensor)

                # Perform classification using ResNet-50
                outputs = resnet_model(result.to(device))
                loss = criterion(outputs, foreground_labels.to(device))

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            else:
                # Perform classification using ResNet-50 without combination
                outputs = resnet_model(foreground_images.to(device))
                loss = criterion(outputs, foreground_labels.to(device))

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    resnet_model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculation
        for foreground_images, foreground_labels in val_loader:  # Load validation images

            # Get masks using BASNet
            d1, d2, d3, d4, d5, d6, d7, d8 = basnet(
                foreground_images.to(device))
            pred = d1[:, 0, :, :]
            masks = normPRED(pred)
            del d1, d2, d3, d4, d5, d6, d7, d8
            masks = masks.detach().cpu()

            # Combine foreground and background with masks
            combined_images = foreground_images * \
                masks + background_images * (1 - masks)

            # Perform classification using ResNet-50
            outputs = resnet_model(combined_images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += foreground_labels.size(0)
            correct += (predicted == foreground_labels.to(device)).sum().item()

    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy:.4f}")
    torch.save(resnet_model.state_dict(), 'resnet_model.pth')
