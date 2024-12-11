import os
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import shutil
import requests
from zipfile import ZipFile

# URL do dataset
dataset_url = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip"
dataset_path = "cats_and_dogs.zip"

# Baixar o dataset
if not os.path.exists(dataset_path):
    print("Baixando o dataset...")
    response = requests.get(dataset_url, stream=True)
    with open(dataset_path, "wb") as file:
        shutil.copyfileobj(response.raw, file)
    print("Dataset baixado!")

# Extrair o dataset
if not os.path.exists("PetImages"):
    print("Extraindo o dataset...")
    with ZipFile(dataset_path, "r") as zip_ref:
        zip_ref.extractall(".")
    print("Dataset extraído!")

# Diretórios para treino
base_dir = "PetImages"
train_dir = "data/train"
os.makedirs(train_dir, exist_ok=True)

def prepare_data():
    for label in ['Cat', 'Dog']:
        label_dir = os.path.join(base_dir, label)
        target_dir = os.path.join(train_dir, label.lower())
        os.makedirs(target_dir, exist_ok=True)
        for img in os.listdir(label_dir):
            if img.endswith('.jpg'):
                shutil.move(os.path.join(label_dir, img), os.path.join(target_dir, img))

prepare_data()


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader

# Configurações
BATCH_SIZE = 32
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformações para as imagens
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Carregar os dados
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Modelo pré-treinado ResNet18
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)  # Saída binária (dog ou cat)
model = model.to(DEVICE)

# Otimizador e função de perda
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Treinar o modelo
EPOCHS = 5
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader)}")

# Salvar o modelo
torch.save(model.state_dict(), "dog_or_cat_model.pth")


from PIL import Image

# Diretório para classificação
classify_dir = "/dogorcat"

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
