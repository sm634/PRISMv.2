import os
from dotenv import load_dotenv
import sys

from utils.files_handler import FileHandler
from pymilvus import connections, utility
from pymilvus import Collection, DataType, FieldSchema, CollectionSchema


class MilvusConnector:

    def __init__(self):
        """
        Zilliz Cloud (Milvus) vector db connector class.
        """
        # AUTHENTICATION
        # store connection keys from .env file by loading it as environment variables.
        load_dotenv()
        # we will use the serverless connection method which only requires the uri and token (API key).
        self.zilliz_uri = os.environ['ZILLIZ_URI']
        self.token = os.environ['ZILLIZ_API_KEY']

        # store the created schema here.
        self.schema = None
        self.server = "default"
        self.collection_name = 'my_collection'
        self.collection = None

        # get params from config for building and searching indexes.
        file_handler = FileHandler()
        file_handler.get_config(config_file_name='vector_db_config')
        self.index_building_params = file_handler.config['MILVUS']['INDEX_BUILDING_PARAMS']
        self.search_params = file_handler.config['MILVUS']['SEARCH_PARAMS']

        # instantiating the class will connect to the server instance of Zilliz cloud cluster specified
        # in the .env file and stored in self.zilliz_uri and self.token attributes.

    def connect_to_db(self):
        connections.connect(self.server,
                            uri=self.zilliz_uri,
                            token=self.token)
        print(f"Connected to DB: {self.zilliz_uri}")

    def disconnect(self):
        connections.disconnect(self.server)

    def set_collection_name(self, collection_name):
        """A function to assign the collection name. Otherwise, it will be called
        default_collection"""
        self.collection_name = collection_name

    def check_collection_exists(self):
        """A simple function that checks if the collection exists in the db and returns a boolean"""
        check_collection = utility.has_collection(self.collection_name)
        return check_collection

    def drop_collection(self, if_exists=True):
        # first connect to DB.
        # check to see if collection exists.
        if if_exists:
            check_collection = self.check_collection_exists()
            if check_collection:
                utility.drop_collection(self.collection_name)
                print(f"Success! collection {self.collection_name} has been dropped.")
            else:
                print(f"No such collection exists in DB {self.zilliz_uri}.")
        else:
            utility.drop_collection(self.collection_name)

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
        fields = [
            FieldSchema(
                name=primary_id,
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True
            ),
            FieldSchema(
                name=field_name,
                dtype=DataType.FLOAT_VECTOR,
                dim=embeddings_dim
            )
        ]
        self.schema = CollectionSchema(
            fields=fields,
            description=f"A Schema for the {self.collection_name} collection",
            enable_dynamic_field=True
        )

    def create_collection(self, shards_num=2, server='default'):
        """
        A function to create a collection.
        :param shards_num: the number of shards to use.
        :param server: the server to create the collection on.
        :param schema: the schema to be used to create the collection.
        """
        if not str(type(self.schema)) == "<class 'pymilvus.orm.schema.CollectionSchema'>":
            print("""Please create a schema before creating a collection.
            This can be done using the create_default_schema method or by assigning your custom
            schema by assigning it to the .schema attribute of your MilvusConnector instance.""")
            raise TypeError

        collection = Collection(
            name=self.collection_name,
            schema=self.schema,
            using=server,
            shards_num=shards_num
        )
        self.collection = collection

    def create_index(self, field_name):
        """
        A function to create index using attributes stored in the class.
        """
        if isinstance(self.collection, type(None)):
            self.create_collection()

        self.collection.create_index(
            field_name=field_name,
            index_params=self.index_building_params
        )

    def search_collection(self, field_name, query_vector, limit=10, expr=None):
        """
        A function to search collection using query vectors.
        :param field_name: name of the vector field to search on.
        :param query_vector: the query vector to search with.
        :param limit: the limit to the number of embeddings retrieved.
        :param expr: Bool, used to filter attributes. See: https://milvus.io/docs/boolean.md
        """
        if isinstance(self.collection, type(None)):
            print("""Please create the collection first using the .create_collection method""")
            sys.exit()

        else:
            self.collection.load()
            results = self.collection.search(
                data=query_vector,
                anns_field=field_name,
                param=self.search_params,
                limit=limit,
                expr=expr
            )
            return results
