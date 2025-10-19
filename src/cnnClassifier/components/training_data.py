import os
import numpy as np
import tensorflow as tf
from cnnClassifier.entity.config_entity import TrainingConfig
from mlflow.models.signature import infer_signature
from pathlib import Path
from cnnClassifier import logger
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        # Load existing model (custom CNN, not VGG)
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
                fill_mode='nearest',
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
        # Compute class weights from generator
        class_indices = self.train_generator.class_indices
        class_counts = {i:0 for i in class_indices.values()}
        # Map subfolder class names to class_idx
        for class_name, class_idx in class_indices.items():
            class_dir = os.path.join(self.config.training_data, class_name)
            if os.path.exists(class_dir):
                # Count files in class subdir only if they're part of the "training" split
                files = [
                    f for f in os.listdir(class_dir)
                    if os.path.isfile(os.path.join(class_dir, f))
                ]
                class_counts[class_idx] = len(files)
        total_samples = sum(class_counts.values())
        num_classes = len(class_counts)
        class_weights = {}
        for class_idx, count in class_counts.items():
            if count == 0:
                logger.warning(f"Class {class_idx} has no training samples! Assigning weight=1")
                class_weights[class_idx] = 1.0
            else:
                class_weights[class_idx] = total_samples / (num_classes * count)
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

        # --- Callbacks for robust training ---
        lr_callback = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1
        )
        # early_stop = EarlyStopping(
        #     monitor='val_loss',
        #     patience=7,
        #     restore_best_weights=True,
        #     verbose=1
        # )
        chkpt = ModelCheckpoint(
            'model_best.h5',
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        )

        history = self.model.fit(
            self.train_generator,
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.config.params_epochs,
            validation_data=self.valid_generator,
            validation_steps=self.validation_steps,
            callbacks=[lr_callback, chkpt],
            class_weight=class_weights
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )
        return history