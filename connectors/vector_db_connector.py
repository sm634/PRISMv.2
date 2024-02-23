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
        # store the created schema here.
        self.schema = {}
        self.server = "default"
        self.collection_name = 'test'

    def set_collection_name(self, collection_name):
        """A function to assign the collection name. Otherwise, it will be called
        default_collection"""
        self.collection_name = collection_name

    def __connect_to_db(self):
        connections.connect(self.server,
                            uri=self.zilliz_uri,
                            token=self.token)
        print(f"Connecting to DB: {self.zilliz_uri}")

    def drop_collect(self):
        # first connect to DB.
        self.__connect_to_db()
        # check to see if collection exists.
        check_collection = utility.has_collection(self.collection_name)
        if check_collection:
            drop_result = utility.drop_collection(self.collection_name)
            print(f"Success! collection {self.collection_name} has been dropped.")
        else:
            print(f"No such collection exists in DB {self.zilliz_uri}")

    def create_default_schema(self,
                              primary_id: str,
                              field_name: str,
                              embeddings_dim: int = 768):
        """
        A function that creates a simple schema only using the primary id and a field
        that that will store the embeddings.
        :param primary_id: The primary id name to be used as primary key for the collection.
        :param field_name: type.FLOAT_VECTOR to store the embeddings.
        :param embeddings_dim: the dimension of the embedding for the field.
        """
        collection_name = "collection"
        if len(self.collection_name) > 0:
            collection_name = self.collection_name

        p_id = FieldSchema(
            name=primary_id,
            dtype=DataType.INT64,
            is_primary=True
        )
        vector_field = FieldSchema(
            name=field_name,
            dtype=DataType.FLOAT_VECTOR,
            dim=embeddings_dim
        )
        self.schema = CollectionSchema(
            fields=[p_id, vector_field],
            description=f"A Schema for the {collection_name} collection",
            enable_dynamic_field=True
        )

    def create_collection(self, shards_num=2, server='default'):
        """
        A function to create a collection.
        :param shards_num: the number of shards to use.
        :param server: the server to create the collection on.
        """
        collection = Collection(
            name=self.collection_name,
            schema=self.schema,
            using=server,
            shards_num=shards_num
        )
        return collection
