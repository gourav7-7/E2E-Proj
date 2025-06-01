import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig



# class PrepareBaseModel:
#     def __init__(self, config: PrepareBaseModelConfig):
#         self.config = config

    
#     def get_base_model(self):
        
#         self.model = tf.keras.applications.vgg16.VGG16(
#             input_shape=self.config.params_image_size,
#             weights=self.config.params_weights,
#             include_top=self.config.params_include_top
#         )

#         self.save_model(path=self.config.base_model_path, model=self.model)

    

#     @staticmethod
#     def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
#         if freeze_all:
#             for layer in model.layers:
#                 model.trainable = False
#         elif (freeze_till is not None) and (freeze_till > 0):
#             for layer in model.layers[:-freeze_till]:
#                 model.trainable = False

#         flatten_in = tf.keras.layers.Flatten()(model.output)
#         dense = tf.keras.layers.Dense(256, activation="relu")(flatten_in)
#         dropout = tf.keras.layers.Dropout(0.5)(dense)
#         prediction = tf.keras.layers.Dense(
#                 units=classes,
#                 activation="softmax"
#                 )(dropout)
#         full_model = tf.keras.models.Model(
#             inputs=model.input,
#             outputs=prediction
#         )

#         full_model.compile(
#             optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
#             loss=tf.keras.losses.CategoricalCrossentropy(),
#             metrics=["accuracy"]
#         )

#         full_model.summary()
#         return full_model
    
    
#     def update_base_model(self):
#         self.full_model = self._prepare_full_model(
#             model=self.model,
#             classes=self.config.params_classes,
#             freeze_all=True,
#             freeze_till=None,
#             learning_rate=self.config.params_learning_rate
#         )

#         self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    
        
#     @staticmethod
#     def save_model(path: Path, model: tf.keras.Model):
#         model.save(path)

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
    
    def get_base_model(self):
        self.model = self._build_custom_cnn()
        self.save_model(path=self.config.base_model_path, model=self.model)
    
    def _build_custom_cnn(self):
        model = tf.keras.models.Sequential()
        
        # Layer 1
        model.add(tf.keras.layers.Conv2D(
            filters=3,
            kernel_size=(3, 3),
            activation='relu',
            input_shape=self.config.params_image_size,
            padding='same'
        ))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        
        # Layer 2
        model.add(tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            activation='relu',
            padding='valid'
        ))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        
        # Layer 3
        model.add(tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation='relu',
            padding='valid'
        ))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        
        # # Layer 4
        # model.add(tf.keras.layers.Conv2D(
        #     filters=64,
        #     kernel_size=(3, 3),
        #     activation='relu',
        #     padding='valid'
        # ))
        # model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        
        # Fully Connected Layers
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(256, activation='relu'))
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        
        # Output Layer - UPDATED to use config classes
        model.add(tf.keras.layers.Dense(
            self.config.params_classes,
            activation='softmax'
        ))
        
        model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=self.config.params_learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        model.summary()
        return model
        
       
    
    def update_base_model(self):
        # No changes needed for custom CNN
        self.save_model(path=self.config.updated_base_model_path, model=self.model)
    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)