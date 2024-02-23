import os
from dotenv import load_dotenv

import time
import random
from pymilvus import connections, utility
from pymilvus import Collection, DataType, FieldSchema, CollectionSchema


class MilvusConnector:

    def __init__(self):
        """
        Zilliz Cloud (Milvus) vector db connector class.
        """
        # store connection keys from .env file by loading it as environment variables.
        load_dotenv()
        # we will use the serverless connection method
        # which only requires the uri and token (API key).
        self.zilliz_uri = os.environ['ZILLIZ_URI']
        self.token = os.environ['ZILLIZ_API_KEY']

    def connect_to_db(self):
        connections.connect("default",
                            uri=self.zilliz_uri,
                            token=self.token)
        print(f"Connecting to DB: {self.zilliz_uri}")


