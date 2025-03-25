from src.log_classifier.utils.classifiers.bert_classifier import bert_classifier
from src.log_classifier.utils.classifiers.llm_classifier import llm_classifier
from src.log_classifier.utils.classifiers.regex_classifier import regex_classifier


def classify(logs):
    labels = []
    for source, log_msg in logs:
        label = log_classifier(source, log_msg)
        labels.append(label)
    return labels


def log_classifier(source, log_msg):
    if source == "LegacyCRM":
        label = llm_classifier(log_msg)
    else:
        label = regex_classifier(log_msg)
        if not label:
            label = bert_classifier(log_msg)
    return label

def csv_classifier(input_file):
    import pandas as pd
    df = pd.read_csv(input_file)

    # Perform classification
    logs = list(zip(df["source"], df["log_message"]))
    df["target_label"] = classify(logs)

    # Save the modified file
    output_file = "output.csv"
    df.to_csv(output_file, index=False)

    return output_file

if __name__ == '__main__':
    csv_classifier("test.csv")
    # logs = [
    #     ("ModernCRM", "IP 192.168.133.114 blocked due to potential attack"),
    #     ("BillingSystem", "User 12345 logged in."),
    #     ("AnalyticsEngine", "File data_6957.csv uploaded successfully by user User265."),
    #     ("AnalyticsEngine", "Backup completed successfully."),
    #     ("ModernHR", "GET /v2/54fadb412c4e40cdbaed9335e4c35a9e/servers/detail HTTP/1.1 RCODE  200 len: 1583 time: 0.1878400"),
    #     ("ModernHR", "Admin access escalation detected for user 9429"),
    #     ("LegacyCRM", "Case escalation for ticket ID 7324 failed because the assigned support agent is no longer active."),
    #     ("LegacyCRM", "Invoice generation process aborted for order ID 8910 due to invalid tax calculation module."),
    #     ("LegacyCRM", "The 'BulkEmailSender' feature is no longer supported. Use 'EmailCampaignManager' for improved functionality."),
    #     ("LegacyCRM", " The 'ReportGenerator' module will be retired in version 4.0. Please migrate to the 'AdvancedAnalyticsSuite' by Dec 2025")
    # ]
    # labels = classify(logs)
    #
    # for log, label in zip(logs, labels):
    #     print(log[0], "->", label)

