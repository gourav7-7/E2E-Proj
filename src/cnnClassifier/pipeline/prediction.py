import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        # Load model
        model = load_model(os.path.join("artifacts", "training", "model.h5"))

        # Load and preprocess image
        imagename = self.filename
        test_image = image.load_img(imagename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        # Predict
        predictions = model.predict(test_image)
        result = np.argmax(predictions, axis=1)[0]

        # Class labels
        class_labels = ['Cyst', 'Normal', 'Stone', 'Tumor']
        
        # Return the predicted class
        prediction = class_labels[result]
        return [{"image": prediction}]
