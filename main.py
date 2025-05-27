from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_02_model_preparation import ModelPreparationPipeline
from cnnClassifier.pipeline.stage_03_model_training import ModelTrainingPipeline




# STAGE_NAME = "Data Ingestion Stage"
# try:
#     logger.info(f">>>>>>>> stage {STAGE_NAME} started here <<<<<<<<")
#     obj = DataIngestionTrainingPipeline()
#     obj.main()
#     logger.info(f">>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<\n\nX=======================================================X")
# except Exception as e:
#     logger.exception(e)
#     raise e

STAGE_NAME = "Model Preparation"
try:
    logger.info(f">>>>>>>>>>>>>> Stage {STAGE_NAME} started <<<<<<<<<<<<<<")
    prepare_base_model = ModelPreparationPipeline()
    prepare_base_model.main()
    logger.info(f">>>>>>>>>>>> Stage {STAGE_NAME} COMPLETED <<<<<<<<<\n\nX=========================================================X")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Model Training"
try:
    logger.info(f">>>>>>>>>>>>>> Stage {STAGE_NAME} started <<<<<<<<<<<<<<")
    training_model = ModelTrainingPipeline()
    training_model.main()
    logger.info(f">>>>>>>>>>>> Stage {STAGE_NAME} COMPLETED <<<<<<<<<\n\nX=========================================================X")
except Exception as e:
    logger.exception(e)
    raise e