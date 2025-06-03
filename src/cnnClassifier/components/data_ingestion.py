import os
import shutil
# import urllib.request as req
import zipfile
import gdown
from cnnClassifier import logger
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self) -> str:
        '''
        fetch data from the url
        '''

        try:
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs("artifacts/data_ingestion", exist_ok=True)
            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            file_id = dataset_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            gdown.download(prefix+file_id, zip_download_dir)

            logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")

        except Exception as e:
            raise e
        



    def extract_zip_file(self):
        '''
        Extracts and reorganizes the zip file to have a flat structure
        '''
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        
        # 1. Extract to a temporary directory
        temp_extract_dir = os.path.join(unzip_path, "temp_extract")
        os.makedirs(temp_extract_dir, exist_ok=True)
        
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(temp_extract_dir)
        
        # 2. Find the actual data directory (handles nested structure)
        data_dir = self._find_data_directory(temp_extract_dir)
        
        if not data_dir:
            raise FileNotFoundError("Could not find class directories in the extracted data")
        
        # 3. Move class directories to the main unzip_path
        for class_name in os.listdir(data_dir):
            class_src = os.path.join(data_dir, class_name)
            class_dest = os.path.join(unzip_path, class_name)
            
            if os.path.isdir(class_src):
                # Remove destination if exists
                if os.path.exists(class_dest):
                    shutil.rmtree(class_dest)
                shutil.move(class_src, class_dest)
                logger.info(f"Moved {class_name} to {class_dest}")
        
        # 4. Clean up temporary files
        shutil.rmtree(temp_extract_dir)
        logger.info("Cleaned up temporary extraction directory")
    
    def _find_data_directory(self, base_path):
        """
        Recursively searches for the directory containing class folders
        """
        # Look for directories that match our expected class names
        expected_classes = {'Normal', 'Tumor', 'Stone', 'Cyst'}
        
        # Check current directory
        if expected_classes.issubset(set(os.listdir(base_path))):
            return base_path
        
        # Check subdirectories
        for root, dirs, files in os.walk(base_path):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                dir_contents = os.listdir(dir_path)
                
                if expected_classes.issubset(set(dir_contents)):
                    return dir_path
        
        return None