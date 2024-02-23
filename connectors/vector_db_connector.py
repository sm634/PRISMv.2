from dotenv import load_dotenv
import time
import random
from pymilvus import connections, utility
from pymilvus import Collection, DataType, FieldSchema, CollectionSchema

import os

if __name__ == '__main__':
    # connect to milvus
    load_dotenv()
    # we will use the serverless connection method which only requires the uri and token (API key).
    zilliz_uri = os.environ['ZILLIZ_URI']
    token = os.environ['ZILLIZ_API_KEY']
