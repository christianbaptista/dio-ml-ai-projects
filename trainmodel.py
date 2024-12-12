import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
from shutil import copyfile
import random


def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    files = []
    for filename in os.listdir(SOURCE):
        file = SOURCE + filename
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " is zero length, so ignoring.")

    training_length = int(len(files) * SPLIT_SIZE)
    testing_length = int(len(files) - training_length)
    shuffled_set = random.sample(files, len(files))
    training_set = shuffled_set[0:training_length]
    testing_set = shuffled_set[-testing_length:]

    for filename in training_set:
        this_file = SOURCE + filename
        destination = TRAINING + filename
        copyfile(this_file, destination)

    for filename in testing_set:
        this_file = SOURCE + filename
        destination = TESTING + filename
        copyfile(this_file, destination)

def TrainModel(train_dir):
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
    print("Treinamento realizado e arquivo salvo!")



# Função para classificar uma imagem
def classify_image(image_path, classify_dir,DEVICE):

    # Carregar modelo salvo
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load("dog_or_cat_model.pth"))
    model = model.to(DEVICE)
    model.eval()

    # Configurações

    IMAGE_SIZE = 224

    # Transformações para as imagens
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

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
