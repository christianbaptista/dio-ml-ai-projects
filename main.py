import prepare



# URL do dataset
dataset_url = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
dataset_path = "cats_and_dogs.zip"
prepare.DownloadDataset(dataset_url=dataset_url, dataset_path=dataset_path)
