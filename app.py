# # from flask import Flask, request, jsonify, render_template
# # import os
# # from flask_cors import CORS, cross_origin
# # from cnnClassifier.utils.common import decodeImage
# # from cnnClassifier.pipeline.prediction import PredictionPipeline



# # os.putenv('LANG', 'en_US.UTF-8')
# # os.putenv('LC_ALL', 'en_US.UTF-8')

# # app = Flask(__name__)
# # CORS(app)


# # class ClientApp:
# #     def __init__(self):
# #         self.filename = "inputImage.jpg"
# #         self.classifier = PredictionPipeline(self.filename)


# # @app.route("/", methods=['GET'])
# # @cross_origin()
# # def home():
# #     return render_template('index.html')




# # @app.route("/train", methods=['GET','POST'])
# # @cross_origin()
# # def trainRoute():
# #     os.system("python main.py")
# #     # os.system("dvc repro")
# #     return "Training done successfully!"



# # @app.route("/predict", methods=['POST'])
# # @cross_origin()
# # def predictRoute():
# #     image = request.json['image']
# #     decodeImage(image, clApp.filename)
# #     result = clApp.classifier.predict()
# #     return jsonify(result)


# # if __name__ == "__main__":
# #     clApp = ClientApp()

# #     app.run(host='0.0.0.0', port=8080) #for AWS

# from flask import Flask, request, jsonify, render_template
# import os
# from flask_cors import CORS, cross_origin
# from cnnClassifier.utils.common import decodeImage
# from cnnClassifier.pipeline.prediction import PredictionPipeline
# import subprocess
# from threading import Thread
# from functools import wraps
# import time

# os.putenv('LANG', 'en_US.UTF-8')
# os.putenv('LC_ALL', 'en_US.UTF-8')

# app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "*"}})

# # Load API secret from environment (set in .env)
# API_SECRET = os.getenv('API_SECRET', 'your-secret-key')

# class ClientApp:
#     def __init__(self):
#         self.filename = "inputImage.jpg"
#         self.classifier = PredictionPipeline(self.filename)
        
# # Training status tracking
# TRAINING_STATUS = {
#     "status": "idle",
#     "progress": 0,
#     "start_time": None,
#     "end_time": None
# }

# # Authentication decorator
# def auth_required(f):
#     @wraps(f)
#     def decorated(*args, **kwargs):
#         auth = request.headers.get('X-API-KEY')
#         if not auth or auth != API_SECRET:
#             return jsonify({"error": "Unauthorized"}), 401
#         return f(*args, **kwargs)
#     return decorated

# @app.route("/", methods=['GET'])
# @cross_origin()
# def home():
#     return render_template('index.html')

# @app.route("/train", methods=['POST'])
# @cross_origin()
# @auth_required
# def trainRoute():
#     def train_task():
#         global TRAINING_STATUS
#         TRAINING_STATUS = {
#             "status": "running",
#             "progress": 0,
#             "start_time": time.time(),
#             "end_time": None
#         }
        
#         stages = [
#             ("Data Ingestion", 20),
#             ("Model Preparation", 40),
#             ("Model Training", 70),
#             ("Model Evaluation", 100)
#         ]
        
#         try:
#             for stage_name, progress in stages:
#                 TRAINING_STATUS["status"] = stage_name
#                 TRAINING_STATUS["progress"] = progress
                
#                 # Run each stage
#                 if stage_name == "Data Ingestion":
#                     subprocess.run(["python", "src/cnnClassifier/pipeline/stage_01_data_ingestion.py"], check=True)
#                 elif stage_name == "Model Preparation":
#                     subprocess.run(["python", "src/cnnClassifier/pipeline/stage_02_model_preparation.py"], check=True)
#                 elif stage_name == "Model Training":
#                     subprocess.run(["python", "src/cnnClassifier/pipeline/stage_03_model_training.py"], check=True)
#                 elif stage_name == "Model Evaluation":
#                     subprocess.run(["python", "src/cnnClassifier/pipeline/stage_04_model_eval.py"], check=True)
            
#             TRAINING_STATUS["status"] = "completed"
#             TRAINING_STATUS["end_time"] = time.time()
            
#         except subprocess.CalledProcessError as e:
#             TRAINING_STATUS["status"] = f"failed: {str(e)}"
#             app.logger.error(f"Training failed: {str(e)}")
    
#     # Start training in background thread
#     if TRAINING_STATUS["status"] not in ["running", "completed"]:
#         thread = Thread(target=train_task)
#         thread.start()
#         return jsonify({"status": "Training started"}), 202
#     else:
#         return jsonify({"status": "Training already in progress or completed"}), 409

# @app.route("/training-status", methods=['GET'])
# @cross_origin()
# def training_status():
#     return jsonify(TRAINING_STATUS)

# @app.route("/predict", methods=['POST'])
# @cross_origin()
# def predictRoute():
#     try:
#         if 'image' not in request.json:
#             return jsonify({"error": "Missing 'image' in request"}), 400
            
#         image = request.json['image']
#         decodeImage(image, clApp.filename)
#         result = clApp.classifier.predict()
#         return jsonify(result)
#     except Exception as e:
#         app.logger.error(str(e))
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     clApp = ClientApp()
#     app.run(host='0.0.0.0', port=8080)

import os
import uuid
import numpy as np
from flask import Flask, render_template, request, jsonify
from cnnClassifier.pipeline.prediction import PredictionPipeline
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    
    # Check if file is selected
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    # Check file extension
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG'}), 400

    try:
        # Save uploaded file with unique name
        filename = f"{uuid.uuid4().hex}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        predictor = PredictionPipeline(filepath)
        result = predictor.predict()
        
        # Clean up temporary file
        os.remove(filepath)
        
        return jsonify({
            'prediction': result['class'],
            'probabilities': result['probabilities']
        })
        
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': 'Error processing image'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)