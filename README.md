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
    * We can also use Few Shot Learning approach to tell the LLM model about the new logs.
  * If there are not enough training samples, then a LLM can be used on these logs.
* For BERT model, we can use **Logistic Regression** for classification.

### Preprocessing
* As part of preprocessing, get the number of logs in each category. Then we can know which category has more logs and which category has less logs.
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