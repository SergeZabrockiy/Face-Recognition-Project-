from Utilities import Utilities
from StackedHourglass import StackedHourglass, CelebAHeatmapDataset
import os
import torch
import numpy as np
from PIL import Image
import math
import matplotlib.pyplot as plt
import cv2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class FaceAlignment:
    def __init__(self, crop_faces, input_size=256, output_size=64, num_stacks = 2, num_blocks = 4, num_classes = 5):

        self.crop_faces = crop_faces
        self.input_size = input_size
        self.output_size = output_size 
        self.num_stacks  = num_stacks  # Hourglass-блоки
        self.num_blocks  = num_blocks  # Residual-блоки в ветвях Hourglass
        self.num_classes = num_classes # глаза, нос, рот
        self.aligned_faces = []

        self.model = StackedHourglass(self.num_stacks, self.num_blocks, self.num_classes).to(device)
        self.model.load_state_dict(
            torch.load(os.path.join('models', 'FaceAlignment.pt')))
        self.dataset = CelebAHeatmapDataset(self.crop_faces, self.input_size, self.output_size)

    def align_face_by_rotation(self, img_idx):
        image_tensor, target_heatmaps, _ = self.dataset[img_idx]
        image_tensor = image_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            predicted_heatmaps = self.model(image_tensor)[-1]
        input_size = self.dataset.input_size
        output_size = self.dataset.output_size
        scale_factor = input_size / output_size
        pred_landmarks = Utilities.post_process_landmarks(predicted_heatmaps, scale_factor)[0]
        img_display = image_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_display = std * img_display + mean
        img_display = np.clip(img_display, 0, 1)

        img_bgr = cv2.cvtColor((img_display * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        aligned_face, dist_zero = Utilities.align_face_by_rotation(img_bgr, pred_landmarks)

        aligned_face_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        
        self.aligned_faces.append(aligned_face_rgb)    
        return dist_zero 

    def show_align_face_by_rotation(self, img_idx):
        plt.subplot(1, 2, 1)
        img_display = self.dataset[img_idx][0].squeeze().cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_display = std * img_display + mean
        img_display = np.clip(img_display, 0, 1)
        plt.imshow(img_display)
        plt.title("Original Face")
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(self.aligned_faces[img_idx])
        plt.title("Aligned Face")
        plt.axis('off')
        plt.show()