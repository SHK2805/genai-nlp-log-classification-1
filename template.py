import os
from pathlib import Path
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')


def create_directory(path: Path):
    if not path.exists():
        os.makedirs(path, exist_ok=True)
        logging.info(f"Creating directory: {path}")
    else:
        logging.info(f"Directory {path} already exists")


def create_file(filepath: Path):
    if not filepath.exists() or filepath.stat().st_size == 0:
        filepath.touch()
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filepath.name} already exists")


def create_project_structure(project_name: str) -> bool:
    try:
        data_folder_name: str = "data"
        list_of_files = [
            # general files
            "template.py",
            "setup.py",
            ".env",
            ".gitignore",
            "requirements.txt",
            "README.md",
            # logs
            f"logs/{project_name}.log",
            # data folder
            f"{data_folder_name}",
            # schema
            f"data_schema/schema.yaml",
            # src folder
            f"src/__init__.py",
            f"src/{project_name}/__init__.py",
            # logging
            f"src/{project_name}/logging/__init__.py",
            f"src/{project_name}/logging/logger.py",
            # exception
            f"src/{project_name}/exception/__init__.py",
            f"src/{project_name}/exception/exception.py",
            # utils
            f"src/{project_name}/utils/__init__.py",
            f"src/{project_name}/utils/delete_directories.py",
            f"src/{project_name}/utils/utils.py",
            f"src/{project_name}/utils/classifiers/classifier.py",
            f"src/{project_name}/utils/classifiers/regex_classifier.py",
            f"src/{project_name}/utils/classifiers/bert_classifier.py",
            f"src/{project_name}/utils/classifiers/llm_classifier.py",
            # constants
            f"src/{project_name}/constants/__init__.py",
            # config
            f"src/{project_name}/config/__init__.py",
            f"src/{project_name}/config/configuration.py",
            # entity
            f"src/{project_name}/entity/__init__.py",
            f"src/{project_name}/entity/config_entity.py",
            f"src/{project_name}/entity/artifact_entity.py",
            # components
            f"src/{project_name}/components/__init__.py",
            f"src/{project_name}/components/data_ingestion.py",
            f"src/{project_name}/components/data_validation.py",
            f"src/{project_name}/components/data_transformation.py",
            f"src/{project_name}/components/model_trainer.py",
            # pipeline
            f"src/{project_name}/pipeline/__init__.py",
            f"src/{project_name}/pipeline/data_ingestion.py",
            f"src/{project_name}/pipeline/data_validation.py",
            f"src/{project_name}/pipeline/data_transformation.py",
            f"src/{project_name}/pipeline/model_trainer.py",
            # main
            "main.py",
            # clean
            "clean.py",
        ]

        for filepath in list_of_files:
            filepath = Path(filepath)
            filedir = filepath.parent

            # Ensure files are treated as files
            if filepath.name in ['.env']:
                create_file(filepath)
                continue

            # Create directories
            create_directory(filedir)

            # Create files if they do not exist or are empty
            if filepath.suffix:  # Check if it's a file (has a suffix)
                create_file(filepath)
            else:
                create_directory(filepath)
    except Exception as e:
        logging.error(f"Error in creating project structure: {e}")
        return False
    return True


if __name__ == "__main__":
    name = "log_classifier"
    create_project_structure(name)
    logging.info(f"Project structure created for {name}")
