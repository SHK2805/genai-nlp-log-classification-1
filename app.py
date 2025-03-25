import os

import pandas as pd
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import FileResponse
from src.log_classifier.utils.classifiers.classifier import classify

app = FastAPI()

@app.post("/classify/")
async def classify_logs(file: UploadFile):
    # Check if the uploaded file is a CSV
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Input file must be a CSV file.")

    try:
        # Load the CSV content into a DataFrame
        df = pd.read_csv(file.file)

        # Validate required columns
        required_columns = {"source", "log_message"}
        if not required_columns.issubset(df.columns):
            raise HTTPException(
                status_code=400,
                detail=f"CSV must contain the following columns: {', '.join(required_columns)}."
            )

        # Classify the logs
        logs = list(zip(df["source"], df["log_message"]))
        df["target_label"] = classify(logs)

        # Save the processed DataFrame to a file
        # create the output folder if it does not exist
        output_folder = "resources"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)

        output_file = f"{output_folder}/output.csv"
        df.to_csv(output_file, index=False)
        return FileResponse(output_file, media_type='text/csv')

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

    finally:
        file.file.close()
