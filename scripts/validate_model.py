import os
import csv
import torch
from torchvision import models
from data_loader import create_dataloader

# Load model weights and set up the model
weights = models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
model = models.detection.fasterrcnn_resnet50_fpn(weights=weights)
num_classes = 6  # Define the number of classes
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# Load the trained model state
model.load_state_dict(torch.load('models/nudity_prevention_model.pth', map_location='cpu'))
model.eval()

# Create the data loader for the validation set
val_dir = 'dataset/validate'
val_loader = create_dataloader(image_dir=val_dir, labels_dir=None, batch_size=1)

# Check if the validation loader is empty
if len(val_loader) == 0:
    print("Validation loader is empty. Please check the validation dataset directory.")
    exit()

# Setup logging
log_file = 'validation_predictions.csv'
with open(log_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Image', 'Box', 'Label', 'Score'])

# Validation loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print("Running validation...")
with torch.no_grad():
    for batch_idx, (images, img_names) in enumerate(val_loader):
        print(f"Processing batch {batch_idx+1}/{len(val_loader)}")
        images = [image.to(device) for image in images]

        # Get predictions
        predictions = model(images)
        
        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            for i, prediction in enumerate(predictions):
                img_name = img_names[i]
                boxes = prediction['boxes'].cpu().numpy()
                labels = prediction['labels'].cpu().numpy()
                scores = prediction['scores'].cpu().numpy()
                
                for box, label, score in zip(boxes, labels, scores):
                    writer.writerow([img_name, box, label, score])

print("Validation completed. Predictions logged to validation_predictions.csv")
