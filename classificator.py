from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn
import os

#configurations
BATCH_SIZE = 32
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Diretório para classificação
classify_dir = "./dogorcat"

# Carregar modelo salvo
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load("dog_or_cat_model.pth"))
model = model.to(DEVICE)
model.eval()

# Transformações para as imagens
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Função para classificar uma imagem
def classify_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(DEVICE)
    output = model(image)
    prediction = torch.sigmoid(output).item()
    return "dog" if prediction > 0.5 else "cat"

# Classificar todas as imagens em `/dogorcat`
for img_name in os.listdir(classify_dir):
    img_path = os.path.join(classify_dir, img_name)
    if img_name.endswith(('.jpg', '.png', '.jpeg')):
        result = classify_image(img_path)
        print(f"Image {img_name}: {result}")
