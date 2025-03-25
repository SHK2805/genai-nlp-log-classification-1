from sentence_transformers import SentenceTransformer
from src.log_classifier.constants import sentence_transformer_model_name

def bert_classifier(log_message,
                    model,
                    model_embedding = SentenceTransformer(sentence_transformer_model_name)):
    if not model or not model_embedding:
        raise ValueError("Model and model_embedding must be provided")
    if not log_message:
        raise ValueError("log_message must be provided")
    embeddings = model_embedding.encode([log_message])
    probabilities = model.predict_proba(embeddings)[0]
    if max(probabilities) < 0.5:
        return "Unclassified"
    predicted_label = model.predict(embeddings)[0]
    return predicted_label


if __name__ == "__main__":
    pass
    # logs = [
    #     "alpha.osapi_compute.wsgi.server - 12.10.11.1 - API returned 404 not found error",
    #     "GET /v2/3454/servers/detail HTTP/1.1 RCODE   404 len: 1583 time: 0.1878400",
    #     "System crashed due to drivers errors when restarting the server",
    #     "Hello World!",
    #     "Multiple login failures occurred on user 6454 account",
    #     "Server A790 was restarted unexpectedly during the process of data transfer"
    # ]
    # for log in logs:
    #     label = bert_classifier(log, None, None)
    #     print(log, "->", label)