import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
import joblib
import requests
import os
import uuid

class ClassificationDL:
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
        self.model_path = 'sonar_lstm_model.h5'
        self.scaler_path = 'scaler.pkl'
        
        self.load_data()
        self.preprocess_data()
        
    def load_data(self):
        self.df = pd.read_csv(self.dataset_url)
        self.X = self.df.iloc[:, :-1]
        self.y = self.df.iloc[:, -1]
        
    def preprocess_data(self):
        self.scaler = StandardScaler()
        self.X_standardized = self.scaler.fit_transform(self.X)
        self.X = self.X_standardized
        self.y = self.y.values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        
    def create_model(self):
        self.model = tf.keras.Sequential()
        if self.task == "classification" and self.hasChanged:
            for arch in self.architecture:
                if arch['layer'] == "Dense" and arch['define_input_shape'] == "true":
                    self.model.add(Dense(arch['neurons'], input_shape=(self.X_train.shape[1], ), activation=arch['activation']))
                elif arch['layer'] == "Dense" and arch['define_input_shape'] == "false":
                    self.model.add(Dense(arch['neurons'], activation=arch['activation']))
                elif arch['layer'] == "Dropout":
                    self.model.add(Dropout(arch['ratio']))
        else:
            self.model.add(Dense(128, input_shape=(self.X_train.shape[1], ), activation="relu"))
            self.model.add(Dropout(0.1))
            self.model.add(Dense(64, activation="relu"))
            self.model.add(Dropout(0.1))
            self.model.add(Dense(32 ,activation="relu"))
            self.model.add(Dense(1, activation='sigmoid'))
        
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
    def train_model(self):
        self.history = self.model.fit(self.X_train, self.y_train, epochs=self.hyperparameters['epochs'], batch_size=self.hyperparameters['batch_size'], validation_split=self.hyperparameters['validation_size'])
        
    def evaluate_model(self):
        y_pred_proba = self.model.predict(self.X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
        self.accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Accuracy: {self.accuracy}")
        
    def save_model(self):
        self.model.save(self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        
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
            
            files = {
                'bucketName': (None, self.bucket_name),
                'files': open(self.scaler_path, 'rb')
            }
            response_scaler = requests.put(self.api_url, files=files)
            response_data_scaler = response_scaler.json()
            scaler_url = response_data_scaler.get('locations', [])[0] if response_scaler.status_code == 200 else None
            
            if scaler_url:
                print(f"Scaler uploaded successfully. URL: {scaler_url}")
            else:
                print(f"Failed to upload scaler. Error: {response_data_scaler.get('error')}")
                return model_url, None
            
            return model_url, scaler_url
            
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {str(e)}")
            return None, None
        
    def execute(self):
        self.create_model()
        self.train_model()
        self.evaluate_model()
        self.save_model()
        model_url, scaler_url = self.upload_files_to_api()
        
        if model_url and scaler_url:
            _id = str(uuid.uuid4())
            model_obj = {
                "modelUrl": model_url,
                "size": os.path.getsize(self.model_path) / (1024 ** 3),
                "id": _id,
                "helpers": [{"scaler": scaler_url}],
                "modelArch": self.architecture,
                "hyperparameters": self.hyperparameters
            }
            return model_obj
        else:
            return None
