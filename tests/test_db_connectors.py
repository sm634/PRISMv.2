import random
import time
from connectors.vector_db_connector import MilvusConnector


def test_milvus_connector():
    """
    Testing the milvus connector for:
    (i) creating collections
    (ii) uploading entities to that collection
    (iii) loading the collection to conduct search.
    """
    milvus_connector = MilvusConnector()
    milvus_connector.connect_to_db()
    # we will create a test collection.
    field_name = 'dummy_embeddings'  # the name of the vector field.
    dim = 64

    print("Creating Collection Schema")
    milvus_connector.set_collection_name('test_collection')
    milvus_connector.drop_collection(if_exists=True)  # drop collection only if it exists.
    milvus_connector.create_default_schema(
        primary_id='primary_key',
        field_name=field_name,
        embeddings_dim=dim
    )  # create a default schema for the collection.
    milvus_connector.create_collection()  # create the collection using the dummy schema.

    print("Inserting dummy data into the collection.")
    # define some parameters.
    nb = 1000
    start = 0  # first primary key id.
    collection = milvus_connector.collection
    schema = milvus_connector.schema
    print(f"Uploaded schema: {schema}")

    # primary_ids = [i for i in range(start, start + nb)]
    vector_field = [[random.random() for _ in range(dim)] for _ in range(nb)]
    entities = [vector_field]  # since we are using auto_id generator to create the default schema, only the vector
    # field needs to be passed.

    t0 = time.time()
    collection.insert(entities)
    ins_rt = time.time() - t0

    print(f"Succeeded inserting in {round(ins_rt, 4)} seconds!")

    print("Flushing..")
    start_flush = time.time()
    collection.flush()
    end_flush = time.time() - start_flush
    print(f"Succeeded in {round(end_flush), 4} seconds!")

    print("Creating Collection Index")
    milvus_connector.create_index(field_name=field_name)  # build index for the collection just created.

    t0 = time.time()
    print("Loading collection...")
    collection.load()
    t1 = time.time()
    print(f"Succeeded in {round(t1 - t0), 4} seconds!")

    print("SEARCH")
    search_vec = [random.random() for _ in range(dim)]
    print(f"Searching vector: {search_vec}")
    results = milvus_connector.search_collection(
        field_name=field_name,
        query_vector=search_vec
    )
    print(f"Result: {results}")

    milvus_connector.disconnect()
