import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class NudityDataset(Dataset):
    def __init__(self, img_dir, labels_dir=None, img_size=(224, 224)):
        self.img_dir = img_dir
        self.labels_dir = labels_dir
        self.img_size = img_size
        self.img_files = [
            f for f in os.listdir(img_dir) 
            if os.path.isfile(os.path.join(img_dir, f)) and (labels_dir is None or os.path.isfile(os.path.join(labels_dir, os.path.splitext(f)[0] + '.txt')))
        ]
        if not self.img_files:
            print(f"No valid image files found in directory: {img_dir}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_file)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to read image file: {img_path}")
            return None, None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.img_size)

        if self.labels_dir:
            label_file = os.path.join(self.labels_dir, os.path.splitext(img_file)[0] + '.txt')
            boxes = []
            labels = []
            if not os.path.isfile(label_file):
                print(f"Missing label file: {label_file}")
                return None, None
            with open(label_file, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    label = int(parts[0])
                    box = [float(x) for x in parts[1:]]
                    if box[2] > box[0] and box[3] > box[1]:
                        labels.append(label)
                        boxes.append(box)

            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

            if boxes.size(0) == 0:
                return self.__getitem__((idx + 1) % len(self))

            transform = transforms.ToTensor()
            image = transform(image)
            return image, {"boxes": boxes, "labels": labels}
        else:
            transform = transforms.ToTensor()
            image = transform(image)
            return image, img_file

def create_dataloader(image_dir, labels_dir=None, batch_size=1, shuffle=False, num_workers=0):
    dataset = NudityDataset(img_dir=image_dir, labels_dir=labels_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader
