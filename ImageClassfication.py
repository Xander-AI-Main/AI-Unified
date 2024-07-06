import requests
import zipfile
import os
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, callbacks
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import image_dataset_from_directory

zip_file_url = "https://idesign-quotation.s3.ap-south-1.amazonaws.com/NO_COMPANYNAME/data.zip"

temp_zip_path = "data.zip"

extract_to = "extracted_files"

response = requests.get(zip_file_url)
with open(temp_zip_path, "wb") as temp_zip_file:
    temp_zip_file.write(response.content)

with zipfile.ZipFile(temp_zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_to)

os.remove(temp_zip_path)

print(f"Files extracted to: {extract_to}")

data_dir = "extracted_files"
batch_size = 16
img_height = 120
img_width = 120

train_ds = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

val_ds = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

AUTOTUNE = tf.data.experimental.AUTOTUNE

class_names = train_ds.class_names
print(f"class_names = {class_names}")

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = len(class_names)

tf.random.set_seed(123)
model = models.Sequential(
    [
        layers.Rescaling(1.0 / 255, input_shape=[img_height, img_width, 3]),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes),
    ]
)

lr = 0.001
model.compile(
    optimizer=tf.keras.optimizers.Adam(lr),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

early_stopping = callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True,
)

epochs = 1
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[early_stopping],
)

model.save("model.h5")

def upload_files_to_api(self):
    try:
        files = {
            'bucketName': (None, self.bucket_name),
            'files': open(self.model_path, 'rb')
        }
        response_model = requests.put(self.api_url, files=files)
        response_data_model = response_model.json()
        model_url = response_data_model.get('locations', [])[0] if response_model.status_code == 200 else None
        
        if model_url:
            print(f"Model uploaded successfully. URL: {model_url}")
        else: 
            print(f"Failed to upload model. Error: {response_data_model.get('error')}")
            return None, None
        
        return model_url
    
    except requests.exceptions.RequestException as e:
            print(f"An error occurred: {str(e)}")
            return None, None