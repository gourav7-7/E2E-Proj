# import os
# import urllib.request as request
# from zipfile import ZipFile
# import tensorflow as tf
# import time
# from pathlib import Path
# from cnnClassifier.entity.config_entity import TrainingConfig
# from mlflow.models.signature import infer_signature
# import numpy as np


# class Training:
#     def __init__(self, config: TrainingConfig):
#         self.config = config

    
#     # def get_base_model(self):
#     #     self.model = tf.keras.models.load_model(
#     #         self.config.updated_base_model_path
#     #     )

#     def get_base_model(self):
#         # Load custom CNN instead of VGG
#         self.model = tf.keras.models.load_model(
#             self.config.updated_base_model_path
#         )

#     def train_valid_generator(self):

#         datagenerator_kwargs = dict(
#             rescale = 1./255,
#             validation_split=0.20
#         )

#         dataflow_kwargs = dict(
#             target_size=self.config.params_image_size[:-1],
#             batch_size=self.config.params_batch_size,
#             interpolation="bilinear"
#         )

#         valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
#             **datagenerator_kwargs
#         )

#         self.valid_generator = valid_datagenerator.flow_from_directory(
#             directory=self.config.training_data,
#             subset="validation",
#             shuffle=False,
#             **dataflow_kwargs
#         )

#         if self.config.params_is_augmentation:
#             train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
#             rotation_range=15,
#             width_shift_range=0.1,
#             height_shift_range=0.1,
#             shear_range=0.1,
#             zoom_range=0.2,
#             horizontal_flip=True,
#             vertical_flip=True,  # Important for medical images
#             fill_mode='nearest'
#             )          
            
#         else:
#             train_datagenerator = valid_datagenerator

#         self.train_generator = train_datagenerator.flow_from_directory(
#             directory=self.config.training_data,
#             class_mode="categorical",
#             subset="training",
#             shuffle=True,
#             **dataflow_kwargs
#         )

    
#     @staticmethod
#     def save_model(path: Path, model: tf.keras.Model):
#     # Create example input
#         input_shape = model.input_shape[1:]  # Skip batch dimension
#         input_example = np.random.rand(1, *input_shape).astype(np.float32)
    
#     # Infer signature
#         signature = infer_signature(input_example, model.predict(input_example))
    
#     # Save model with signature
#         model.save(path, signatures=signature)

#     def get_class_weights(self):
#         class_counts = np.sum(self.train_generator.labels, axis=0)
#         total = np.sum(class_counts)
#         return {i: total/(4 * count) for i, count in enumerate(class_counts)}



    
#     def train(self):
#         self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
#         self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size
#         class_weights = self.get_class_weights()

#         self.model.fit(
#             self.train_generator,
#             epochs=self.config.params_epochs,
#             steps_per_epoch=self.steps_per_epoch,
#             validation_steps=self.validation_steps,
#             validation_data=self.valid_generator,
#             class_weight=class_weights
#         )

#         self.save_model(
#             path=self.config.trained_model_path,
#             model=self.model
#         )

import os
import numpy as np
import tensorflow as tf
from cnnClassifier.entity.config_entity import TrainingConfig
from mlflow.models.signature import infer_signature
import urllib.request as request
from zipfile import ZipFile
import time
from pathlib import Path
from cnnClassifier import logger



class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    def train_valid_generator(self):
        datagenerator_kwargs = dict(
            rescale=1./255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.1,
                zoom_range=0.2,
                horizontal_flip=True,
                vertical_flip=True,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            class_mode="categorical",
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

    def get_class_weights(self):
    # Get class indices from generator
        class_indices = self.train_generator.class_indices
        class_counts = {}
    
        # Count files in each class directory
        for class_name, class_idx in class_indices.items():
            class_dir = os.path.join(self.config.training_data, class_name)
            if os.path.exists(class_dir):
                # Get all files in the class directory
                files = [f for f in os.listdir(class_dir) 
                     if os.path.isfile(os.path.join(class_dir, f))]
                class_counts[class_idx] = len(files)
    
        # Calculate total samples and number of classes
        total_samples = sum(class_counts.values())
        num_classes = len(class_counts)
    
        # Handle zero counts and calculate weights
        class_weights = {}
        for class_idx, count in class_counts.items():
            if count == 0:
                logger.warning(f"Class {class_idx} has no samples! Using weight=1")
                class_weights[class_idx] = 1.0
            else:
                # Weight formula: total_samples / (num_classes * class_count)
                class_weights[class_idx] = total_samples / (num_classes * count)
    
        # Log the calculated weights
        logger.info(f"Class counts: {class_counts}")
        logger.info(f"Class weights: {class_weights}")
    
        return class_weights

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        input_shape = model.input_shape[1:]
        input_example = np.random.rand(1, *input_shape).astype(np.float32)
        signature = infer_signature(input_example, model.predict(input_example))
        model.save(path, signatures=signature)

    def train(self):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size
        
        class_weights = self.get_class_weights()
        
        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
            class_weight=class_weights
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )