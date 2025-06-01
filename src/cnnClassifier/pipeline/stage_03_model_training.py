# stage_03_model_training.py
from dotenv import load_dotenv
import os

# 1. Load environment variables FIRST
load_dotenv()  # Loads from .env file in project root

# 2. Initialize MLflow/DagsHub connection
import mlflow
import dagshub

dagshub.init(
    repo_owner="gourav7-7", 
    repo_name="E2E-Proj", 
    mlflow=True
)
mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

# 3. Now import other project modules
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.training_data import Training
from cnnClassifier import logger


STAGE_NAME = "Model Training Stage"
class ModelTrainingPipeline:
    def __init__(self):
        pass
        
    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model()
        training.train_valid_generator()
        training.train()



if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>> stage {STAGE_NAME} started here <<<<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<\n\nX=======================================================X")
    except Exception as e:
        logger.exception(e)
        raise e