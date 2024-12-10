import time
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import cv2
from PIL import Image
import numpy as np

# Define the same transformations as before
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # target_size
    transforms.ToTensor(),          # Convert the image to a tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Simplified normalization
])

class CustomResNet50(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomResNet50, self).__init__()
        self.conv_base = models.resnet50(pretrained=True)
        for param in self.conv_base.parameters():
            param.requires_grad = False

        self.conv_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        num_ftrs = self.conv_base.fc.in_features
        self.conv_base.fc = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])
        x = self.conv_base(x)
        return x

# Instantiate the model
model = CustomResNet50()

# Load the trained weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('best_cat_dog.pth', map_location=device)
model.eval()
model.to(device)

def get_prediction_class(prediction_tensor):
    if prediction_tensor[0][1] > prediction_tensor[0][0]:
        return "dog"
    else:
        return "cat"

# Access the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access the webcam.")
    exit()

# Initialize timing variables for FPS calculation
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from webcam.")
        break

    # Convert the frame (BGR) to RGB and then to PIL Image
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)

    # Apply the same transformations used during training and validation
    img_tensor = val_transforms(pil_img).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        output = model(img_tensor)
    predicted_class = get_prediction_class(output)

    # Calculate FPS
    current_time = time.time()
    elapsed_time = current_time - prev_time
    prev_time = current_time
    fps = 1.0 / elapsed_time if elapsed_time > 0 else 0.0

    # Display the result and FPS on the frame
    cv2.putText(frame, f"Predicted: {predicted_class}", (10,30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.putText(frame, f"FPS: {fps:.2f}", (10,70), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.imshow("Webcam Classification", frame)

    # Press 'q' to quit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
