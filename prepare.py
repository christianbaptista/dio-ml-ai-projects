
import requests
import zipfile
import os
import shutil


def DownloadDataset(dataset_url, dataset_path):
    response = requests.get(dataset_url)
    with open(dataset_path, 'wb') as new_file:
        new_file.write(response.content)
    print("Download realizado! Arquivo salvo em : {}".format(dataset_path))

def OpenFileZip(local_zip):
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('/tmp')
    zip_ref.close()
    print("ZIP extracted")


def PrepareData(base_dir, train_dir):
    print("Dados sendo Organizados")
    os.makedirs(train_dir, exist_ok=True)
    for label in ['Cat', 'Dog']:
            label_dir = os.path.join(base_dir, label)
            target_dir = os.path.join(train_dir, label.lower())
            os.makedirs(target_dir, exist_ok=True)
            for img in os.listdir(label_dir):
                if img.endswith('.jpg'):
                    shutil.move(os.path.join(label_dir, img), os.path.join(target_dir, img))

    print("Dados preparados")
