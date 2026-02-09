import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import pandas as pd
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

class ResNetWithArcFace(nn.Module):
    def __init__(self, identifity_path, image_path, num_classes=10177, embedding_size=512, s=30, m=0.5):
        super(ResNetWithArcFace, self).__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, embedding_size).to(device)
        self.arc_face = ArcFaceLoss(embedding_size, num_classes, s=s, m=m).to(device)
        self.identifity_path = identifity_path
        self.image_path = image_path

    def forward(self, x, labels=None):
        features = self.backbone(x)
        original_logits = F.linear(F.normalize(features), F.normalize(self.arc_face.weight)) * self.arc_face.s
        if labels is None:  
            return original_logits
        else: 
            penalized_logits = self.arc_face(features, labels)
            return penalized_logits, original_logits
    
    def predict(self, x, idx, show=False): 
        labels = pd.read_csv(self.identifity_path,  header=None)
        label = labels[labels.iloc[:, 2] == idx].values.tolist()[0]
        x = x.unsqueeze(0)
        x = x.to(device)
        self.eval()                    
        with torch.no_grad():
            features = self.backbone(x)
            embeddings = F.normalize(features, p=2, dim=1)
            w_norm = F.normalize(self.arc_face.weight, p=2, dim=1, eps=1e-12)
            logits = F.linear(embeddings, w_norm) * self.arc_face.s
            predicted_classes = torch.argmax(logits, dim=1)
        if show:
            image = plt.imread(os.path.join(self.image_path, label[0]))
            plt.imshow(image)
            plt.axis('off')
            plt.title(f'{label[0]} {label[1]} {predicted_classes.item()}')
            plt.show()
        return label[0], label[1], predicted_classes

    def get_embeddings(self, x): 
            self.eval()                    
            x = x.unsqueeze(0)
            x = x.to(device)    
            with torch.no_grad():
                features = self.backbone(x)
                embeddings = F.normalize(features, p=2, dim=1)
            return embeddings

class ArcFaceLoss(nn.Module):
    """
    ArcFace слой
    """
    def __init__(self, in_features, out_features, s=30.0, m=0.5):
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = nn.Parameter(torch.tensor(s, dtype=torch.float32))
        self.m = torch.tensor(m)
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = torch.cos(self.m)
        self.sin_m = torch.sin(self.m)
        self.th = torch.cos(torch.pi - self.m)
        self.mm = torch.sin(torch.pi - self.m) * self.m

    def forward(self, input, label):
        x_norm = F.normalize(input, p=2, dim=1, eps=1e-12)
        w_norm = F.normalize(self.weight, p=2, dim=1, eps=1e-12)
        
        cosine = F.linear(x_norm, w_norm)  # cosθ

        # Защита cosθ от выхода за [-1, 1]
        cosine_clipped = torch.clamp(cosine, -1.0, 1.0)
        sine = torch.sqrt(1.0 - cosine_clipped.pow(2))  # sinθ

        # Вычисление cos(θ + m)
        cos_m = torch.cos(self.m)
        sin_m = torch.sin(self.m)
        phi = cosine_clipped * cos_m - sine * sin_m  # cos(θ + m)
        
        # Финальное ограничение phi
        phi = torch.clamp(phi, -1.0, 1.0)
            
        one_hot = torch.zeros(cosine.size(), device=device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

class CelebAHeatmapDataset(Dataset):
    def __init__(self, aligned_faces):
        self.to_pil = transforms.ToPILImage()
        self.aligned_faces = aligned_faces
        self.transform = transforms.Compose([
                                            transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                            ])
    def __len__(self):
        return len(self.aligned_faces)

    def __getitem__(self, idx):
        image = self.to_pil(self.aligned_faces[idx])

        image_tensor = self.transform(image)
        return image_tensor