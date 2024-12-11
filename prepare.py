import requests



def DownloadDataset(dataset_url, dataset_path):
    response = requests.get(dataset_url)
    with open(dataset_path, 'wb') as new_file:
        new_file.write(response.content)
    print("Download realizado! Arquivo salvo em : {}".format(dataset_path))
