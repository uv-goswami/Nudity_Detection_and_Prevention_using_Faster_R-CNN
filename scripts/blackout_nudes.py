import os
import cv2
import torch
import numpy as np
from torchvision import models
from data_loader import create_dataloader

# Load model weights and set up the model
weights = models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
model = models.detection.fasterrcnn_resnet50_fpn(weights=weights)
num_classes = 6  
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

model.load_state_dict(torch.load('models/nudity_prevention_model.pth', weights_only=True))
model.eval()

def blackout_nudes(image):
    input_image = image.unsqueeze(0)
    with torch.no_grad():
        detections = model(input_image)[0]

    confidence_threshold = 0.3
    blackout_applied = False

    for box, label, score in zip(detections['boxes'], detections['labels'], detections['scores']):
        if score >= confidence_threshold and label in [1, 2, 3, 4, 5]:  # Check for relevant labels
            image = torch.zeros_like(image)  # Blackout the entire image
            blackout_applied = True
            break
    return image, blackout_applied

def clear_output_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

# Define directories
test_dir = 'dataset/test'
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# Clear the output directory
clear_output_directory(output_dir)

# Create dataloader
batch_size = 1
dataloader = create_dataloader(image_dir=test_dir, labels_dir=None, batch_size=batch_size)

# Process all images in the test directory
for images, img_names in dataloader:
    for i in range(images.size(0)):
        image = images[i]
        img_name = img_names[i]

        blackout_image, blackout_applied = blackout_nudes(image)

        # Convert tensor to numpy array for saving
        blackout_image = blackout_image.permute(1, 2, 0).cpu().numpy()
        blackout_image = (blackout_image * 255).astype(np.uint8)

        # Display the result
        window_name = 'Nude Image' if blackout_applied else 'Non-Nude Image'
        cv2.imshow(window_name, blackout_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Save the blackout image
        output_path = os.path.join(output_dir, 'blackout_' + os.path.basename(img_name))
        cv2.imwrite(output_path, blackout_image)
        
        if not os.path.exists(output_path):
            print(f"Failed to save image to {output_path}")
        else:
            print(f"Image saved successfully to {output_path}")
        
        print(f"Processed image from {img_name}")

print("Test successfull.")
