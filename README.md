# GenAI NLP Log Classification

### Technologies used
* Python
* LLM
* BERT
* Logistic Regression (Scikit-learn)
* Regular Expressions
* FastAPI
  * For building the server backend

### Problem Statement
* This application acts as a **Log Classifier** which classifies the logs into different categories based on the log message.
* Then the classified logs can be fed into the log management system for further analysis.
* The logs can be of different types/levels like **Security**, **Workflow Error**, **User**, **Resource Usage** etc.
* The logs are obtained from various sources like **Application Logs**, **Server Logs**, **System Logs** etc.
* All the logs can be aggregated into a single system/file and then classified based on the log message.

### Data
* The data used for training the model is a **Synthetic Dataset** which is generated.
* Columns
  * timestamp
  * source
  * log_message
  * target_label
* Here the **log_message** is the input feature (X) and **target_label** is the output feature (y).

### Approach
* The data can have common patterns like **IP Address**, **URL**, **File Path**, **Error Codes** etc.
* But we need to deal with a lot of data. 
* To detect patterns in the log messages, we can use **BERT** model, which is a transformer model.
  * We can use **ANY** model for embedding the log messages.
* The BERT model will convert the log messages into embeddings which can be used for classification.
* The logs are clustered into different categories using **DBSCAN** algorithm.
  * This is an unsupervised learning approach.
* Based on these clusters, we can build regular expressions for each cluster.
* These regular expressions can be used to classify the logs into different categories.
* Any logs that are not classified falls into the **Unknown** category.
* The Unknown category logs can be further classified using the below
  * If there are enough training samples, then a BERT model can be trained on these logs.
    * We can also use a Few Shot Learning approaches to tell the LLM model about the new logs.
  * If there are not enough training samples, then a LLM can be used on these logs.
* For BERT model, we can use **Logistic Regression** for classification.

### What BERT Is:
* BERT, short for **Bidirectional Encoder Representations from Transformers**, is a groundbreaking model in Natural Language Processing (NLP) developed by Google. 
* It is based on the **Transformer architecture**, which has revolutionized the field of NLP. Here's a breakdown:
  * **Bidirectional**: Unlike earlier models, BERT considers context from both directions (left-to-right and right-to-left) when processing text. This helps it understand the meaning of words based on their surrounding context.
  * **Pre-trained**: BERT comes pre-trained on a vast corpus of text (like books and Wikipedia). This enables it to grasp nuances in language before being fine-tuned for specific tasks.
  * **Encoder**: It focuses on encoding input text to extract meaningful representations that can be used for various applications.

### What BERT Is Used For:
* BERT is a versatile NLP model used for tasks that involve understanding and processing human language. Some common applications include:
  1. **Text Classification**: Categorizing text into predefined groups, such as spam detection or sentiment analysis.
  2. **Named Entity Recognition (NER)**: Identifying and extracting entities like names, dates, and locations in text.
  3. **Question Answering**: Providing answers to questions based on a given passage of text (e.g., in search engines or chatbots).
  4. **Text Summarization**: Generating concise summaries from longer pieces of text.
  5. **Language Translation**: Translating text from one language to another.
  6. **Text Similarity**: Determining how similar two pieces of text are useful in clustering and recommendation systems.
  7. **Contextual Embeddings**: Generating rich vector representations for words or sentences to be used in downstream ML tasks.

* BERT's ability to understand deep linguistic context has set a high standard in NLP, influencing many later models, such as RoBERTa and DistilBERT.

### Preprocessing
* As part of preprocessing, get the number of logs in each category. Then we can know which category has more logs and which category has fewer logs.
* Based on the number of logs, we can determine if we need to use a BERT model or a LLM model.

### Code
#### Coding steps
* Constants
* Configuration
* Entity
  * Config Entity
* Entity
  * Artifact Entity
* Components
* Pipeline
* Main

#### Code details
* The constants are defined in the **src/log_classifier/constants/__init__.py** file.

## Coding Files
### Data Ingestion
* For data ingestion we read the data from the `data/synthetic_logs.csv` file
* **Step1**: Add constants to `constants/__init__.py` file
* **Step2**: Add **GENERAL** and **DATA INGESTION** constants to `constants/__init__.py` file
* **Step3**: Add **TrainingPipelineConfig** class to `config/configuration.py` file
* **Step4**: Add **DataIngestionConfig** class to `entity/config_entity.py` file
  * In here we create `DataIngestionConfig` class with the below attributes as class variables
  * These variables contain the folder and file paths for the data ingestion process
  * Using the above variables, we create the below file structure
  ```plaintext
  artifacts/
  └── data_ingestion/
      ├── feature_store/
      │   └── synthetic_logs.csv
      └── ingested/
          └── train_data.csv
  ```
* **Step5**: Add **DataIngestionArtifact** class to `entity/artifact_entity.py` file with paths to train data
* **Step6**: Add **DataIngestion** class to `components/data_ingestion.py` file
  * In here we create `DataIngestion` class
* **Step7**: Add **DataIngestion** class to `pipeline/data_ingestion.py` file
* **Step8**: Add the pipeline to the `main.py` file and run the pipeline