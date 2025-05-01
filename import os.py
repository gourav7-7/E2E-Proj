import os
from pathlib import Path
import logging # -> to get the messages on terminal about file creeation and structure

#logging string this is a replacement for print statment as a better practice.
logging.basicConfig(level=logging.INFO, format='[%a(sctime)s]: %(message)s:')

proj_name = 'cnnClassifier'

list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{proj_name}/components/__init__.py",
    f"src/{proj_name}/utils/__init__.py",
    f"src/{proj_name}/config/__init__.py",
    f"src/{proj_name}/config/configuration.py",
    f"src/{proj_name}/pipeline/__init__.py",
    f"src/{proj_name}/entity/__init__.py",
    f"src/{proj_name}/constant/__init__.py",
    "config/config.ymal",
    "dvc.ymal",
    "params.ymal",
    "requirements.txt",
    "setup.py",
    "research/trails.ipynb",
    "templates/index.html",
    
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating Directory; {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")

    else:
        logging.info(f"{filename} is already exists")

 