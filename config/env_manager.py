import os
from dotenv import load_dotenv
load_dotenv()

# this reads the data from the .env file and returns the value of the key
def get_key(key_name):
    return os.getenv(key_name)

def groq_api_key():
    return get_key('GROQ_API_KEY')