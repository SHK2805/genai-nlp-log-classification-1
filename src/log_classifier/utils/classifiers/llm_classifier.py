import sys

from groq import Groq
import re

from config.set_config import Config
from src.log_classifier.exception.exception import CustomException
from src.log_classifier.logging.logger import logger

try:
    config = Config()
    if config.set():
        logger.info("Environment variables set")
    else:
        logger.error("Environment variables NOT set")
        raise CustomException("Environment variables NOT set", sys)
except Exception as ex:
    logger.error(f"Error running the pipeline: {ex}")
    raise CustomException(ex, sys)

groq = Groq()

def llm_classifier(log_msg):
    """
    Generate a variant of the input sentence. For example,
    If input sentence is "User session timed out unexpectedly, user ID: 9250.",
    variant would be "Session timed out for user 9251"
    """
    prompt = f'''Classify the log message into one of these categories: 
    (1) Workflow Error, (2) Deprecation Warning.
    If you can't figure out a category, use "Unclassified".
    Put the category inside <category> </category> tags. 
    Log message: {log_msg}'''

    chat_completion = groq.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        # model="llama-3.3-70b-versatile",
        model="deepseek-r1-distill-llama-70b",
        temperature=0.5
    )

    content = chat_completion.choices[0].message.content
    match = re.search(r'<category>(.*)<\/category>', content, flags=re.DOTALL)
    category = "Unclassified"
    if match:
        category = match.group(1)

    return category


if __name__ == "__main__":
    print(llm_classifier(
        "Case escalation for ticket ID 7324 failed because the assigned support agent is no longer active."))
    print(llm_classifier(
        "The 'ReportGenerator' module will be retired in version 4.0. Please migrate to the 'AdvancedAnalyticsSuite' by Dec 2025"))
    print(llm_classifier("System reboot initiated by user 12345."))