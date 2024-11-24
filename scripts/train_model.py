import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from data_loader import NudityDataset

weights = models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
model = models.detection.fasterrcnn_resnet50_fpn(weights=weights)
num_classes = 6 
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)


def custom_collate(batch):
    images = torch.stack([torch.tensor(item[0]) for item in batch])
    targets = [{"boxes": item[1]["boxes"], "labels": item[1]["labels"]} for item in batch]
    return images, targets


def validate_data(images, targets):
    for img, target in zip(images, targets):
        if torch.isnan(img).any() or torch.isnan(target["boxes"]).any() or torch.isnan(target["labels"]).any():
            return False
        if torch.isinf(img).any() or torch.isinf(target["boxes"]).any() or torch.isinf(target["labels"]).any():
            return False
    return True


def train_model(dataset, model, optimizer, num_epochs=25):
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=custom_collate)
    model.train()
    for epoch in range(num_epochs):
        print(f'Starting epoch {epoch + 1}/{num_epochs}')
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = list(image.permute(2, 0, 1).float() for image in images)
            if not validate_data(images, targets):
                print(f'Skipping batch {batch_idx} due to invalid data.')
                continue
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {losses.item()}')
        print(f'Completed epoch {epoch + 1}/{num_epochs}')

    print('Training complete.')


if __name__ == "__main__":
    dataset = NudityDataset(img_dir='dataset/train', labels_dir='dataset/labels')
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    train_model(dataset, model, optimizer, num_epochs=25)
    torch.save(model.state_dict(), 'models/nudity_prevention_model.pth')
