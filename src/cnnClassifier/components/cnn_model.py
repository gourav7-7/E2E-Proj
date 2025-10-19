import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self, config):
        self.config = config

    def get_base_model(self):
        self.model = self._build_custom_cnn()
        self.save_model(path=self.config.base_model_path, model=self.model)

    def _build_custom_cnn(self):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        print(gpus)
        for gpu in gpus: 
            tf.config.experimental.set_memory_growth(gpu, True)

        model = tf.keras.models.Sequential()
        # Input normalization
        model.add(tf.keras.layers.Rescaling(1./255, input_shape=self.config.params_image_size))
        
        # Block 1
        model.add(tf.keras.layers.Conv2D(16, (3,3), padding='same', use_bias=False))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('relu'))
        
        # Block 2
        model.add(tf.keras.layers.Conv2D(32, (3,3), padding='same', use_bias=False))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.MaxPooling2D())
        
        # Block 3
        model.add(tf.keras.layers.Conv2D(64, (3,3), padding='same', use_bias=False))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.MaxPooling2D())
        
        # Block 4
        model.add(tf.keras.layers.Conv2D(128, (3,3), padding='same', use_bias=False))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.MaxPooling2D())
        
        # Feature aggregation
        model.add(tf.keras.layers.GlobalAveragePooling2D())
        
        # Dense head
        model.add(tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.04)))
        model.add(tf.keras.layers.Dropout(0.4))
        model.add(tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.04)))
        model.add(tf.keras.layers.Dropout(0.4))
        model.add(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.04)))
        model.add(tf.keras.layers.Dropout(0.4))
        model.add(tf.keras.layers.Dense(64 , activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.03)))
        model.add(tf.keras.layers.Dropout(0.4))
        model.add(tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(16, activation='relu'))
        
        # Output layer
        model.add(tf.keras.layers.Dense(self.config.params_classes, activation='softmax'))
        
        # Optimizer setup
        initial_lr = self.config.params_learning_rate
        try:
            from tensorflow_addons.optimizers import AdamW
            optimizer = AdamW(learning_rate=initial_lr, weight_decay=1e-4)
        except ImportError:
            optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)

        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
            metrics=["accuracy"]
        )
        model.summary()
        return model

    def update_base_model(self):
        self.save_model(path=self.config.updated_base_model_path, model=self.model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)