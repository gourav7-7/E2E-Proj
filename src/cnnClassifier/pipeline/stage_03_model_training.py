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