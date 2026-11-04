import os
from dotenv import load_dotenv

load_dotenv()

def get_pipeconeapi_env():
    """
    Read Pinecone API key from .env file.
    """
    api_key = os.getenv("PINECONE_API_KEY")

    if not api_key:
        raise ValueError("ls is missing from .env file")
    return api_key
if __name__=='__main__':
    a=get_pipeconeapi_env()
    print(a)
