from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.cnn_model import PrepareBaseModel
from cnnClassifier import logger

STAGE_NAME = "Model Preparation Stage"
class ModelPreparationPipeline:
    def __init__(self):
        pass
        
    def main(self):
        config = ConfigurationManager()
        base_model_config = config.get_prepare_base_model_config()
        base_model = PrepareBaseModel(config=base_model_config)
        base_model.get_base_model()
        base_model.update_base_model()
        


if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>> stage {STAGE_NAME} started here <<<<<<<<")
        obj = ModelPreparationPipeline()
        obj.main()
        logger.info(f">>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<\n\nX=======================================================X")
    except Exception as e:
        logger.exception(e)
        raise e