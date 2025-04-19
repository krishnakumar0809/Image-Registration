from utils.registration import register_images
from models.unet import UNet
from utils.dataset import load_images, preprocess
import torch
import torchvision.transforms as transforms
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def segment_image(model, image_tensor):
    model.eval()
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0))
        prediction = torch.sigmoid(output).squeeze().cpu().numpy()
        return (prediction > 0.5).astype(np.uint8)

def main():
    fixed_path = "data/fixed/sample1.png"
    moving_path = "data/moving/sample1.png"

    # Step 1: Register the moving image to the fixed one
    registered_image = register_images(fixed_path, moving_path)
    registered_image.save("outputs/registered.png")

    # Step 2: Preprocess for segmentation
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    input_tensor = transform(registered_image)

    # Step 3: Load U-Net model
    model = UNet(in_channels=1, out_channels=1)
    model.load_state_dict(torch.load("models/unet.pth", map_location="cpu"))

    # Step 4: Segment
    mask = segment_image(model, input_tensor)
    plt.imsave("outputs/segmentation_mask.png", mask, cmap="gray")

if __name__ == "__main__":
    main()
