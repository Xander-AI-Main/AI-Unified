import requests
import zipfile
import os
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, callbacks
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import image_dataset_from_directory
import uuid

class ImageModelTrainer:
    def __init__(self, dataset_url, hasChanged, task, mainType, archType, architecture, hyperparameters):
        self.dataset_url = dataset_url
        self.hasChanged = hasChanged
        self.task = task
        self.mainType = mainType
        self.archType = archType
        self.architecture = architecture
        self.hyperparameters = hyperparameters
        
        self.api_url = 'https://s3-api-uat.idesign.market/api/upload'
        self.bucket_name = 'idesign-quotation'
        self.data_dir = "extracted_files"
        self.img_height = 120
        self.img_width = 120

        self.download_and_extract_data()
        self.prepare_datasets()
        self.build_model()

    def download_and_extract_data(self):
        temp_zip_path = "data.zip"
        
        response = requests.get(self.dataset_url)
        with open(temp_zip_path, "wb") as temp_zip_file:
            temp_zip_file.write(response.content)

        with zipfile.ZipFile(temp_zip_path, "r") as zip_ref:
            zip_ref.extractall(self.data_dir)

        os.remove(temp_zip_path)
        print(f"Files extracted to: {self.data_dir}")

    def prepare_datasets(self):
        train_ds = image_dataset_from_directory(
            self.data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=hyperparameters["batch_size"],
        )

        val_ds = image_dataset_from_directory(
            self.data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=hyperparameters["batch_size"],
        )

        AUTOTUNE = tf.data.experimental.AUTOTUNE

        self.class_names = train_ds.class_names
        print(f"class_names = {self.class_names}")

        self.train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        self.val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    def build_model(self):
        self.num_classes = len(self.class_names)

        tf.random.set_seed(123)
        self.model = models.Sequential(
            [
                layers.Rescaling(1.0 / 255, input_shape=[self.img_height, self.img_width, 3]),
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(32, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dense(128, activation="relu"),
                layers.Dense(self.num_classes),
            ]
        )

        lr = 0.001
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(lr),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

    def train_model(self, epochs):
        early_stopping = callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
        )

        self.history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs,
            callbacks=[early_stopping],
        )

        self.model.save("model.h5")

    def upload_files_to_api(self):
        try:
            files = {
                'bucketName': (None, self.bucket_name),
                'files': open("model.h5", 'rb')
            }
            response_model = requests.put(self.api_url, files=files)
            response_data_model = response_model.json()
            model_url = response_data_model.get('locations', [])[0] if response_model.status_code == 200 else None
            
            if model_url:
                print(f"Model uploaded successfully. URL: {model_url}")
                return model_url
            else: 
                print(f"Failed to upload model. Error: {response_data_model.get('error')}")
                return None
        
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {str(e)}")
            return None

    def execute(self):
        self.train_model(hyperparameters["epochs"])
        model_url = self.upload_files_to_api()
        
        if model_url:
            _id = str(uuid.uuid4())
            model_obj = {
                "modelUrl": model_url,
                "size": os.path.getsize("model.h5") / (1024 ** 3),  # size in GB
                "id": _id,
                "modelArch": self.architecture,
                "hyperparameters": self.hyperparameters
            }
            return model_obj
        else:
            return None

if __name__ == "__main__":
    dataset_url = "https://idesign-quotation.s3.ap-south-1.amazonaws.com/NO_COMPANYNAME/data.zip"
    hasChanged = True
    task = "image"
    mainType = "DL"
    archType = "default" 
    architecture = {} # extract from arch.json
    hyperparameters = {}  # extract from arch.json

    import json

    def returnArch (data, task, mainType, archType):
        current_task = data[task]

        for i in current_task:
            if  i["type"] == mainType and i["archType"] == archType:
                return i["architecture"], i["hyperparameters"]

    arch_data = {}
    task = "text"
    mainType='topic classification'
    archType='default'

    with open ('arch.json', 'r') as f:
        arch_data = json.load(f)

    architecture, hyperparameters = returnArch(arch_data, task, mainType, archType)

    trainer = ImageModelTrainer(dataset_url, hasChanged, task, mainType, archType, architecture, hyperparameters)
    model_obj = trainer.execute()

    if model_obj:
        print(f"Model Object: {model_obj}")
    else:
        print("Failed to train and upload the model.")
