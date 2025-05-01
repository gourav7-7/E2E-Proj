from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline


STAGE_NAME = "DAta Ingestion Stage"
try:
    logger.info(f">>>>>>>> stage {STAGE_NAME} started here <<<<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<\n\nX=======================================================X")
except Exception as e:
    logger.exception(e)
    raise e