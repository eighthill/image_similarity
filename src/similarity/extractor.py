import torch
from torchvision import models
from torchvision import transforms
from PIL import Image

model = models.resnet50(weights="DEFAULT")
model.fc = torch.nn.Identity()
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def extract_embedding(image):

    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        embedding = model(tensor)

    return embedding.squeeze().numpy()