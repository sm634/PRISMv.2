"""
Docstring
---------

This is a set of helper functions to parse responses from vector database Milvus.

Pseudocode:
1. Provide the option to upload collections (if collection already doesn't exist in the DB).
    - create embeddings of the collections you want to upload as documents.
    - insert that to the db.
2. Conduct a similarity search and get back text/passage(s) based on the query that is used to bring back the search
results.
"""
import time
from utils.embedding_funcs import create_embedding, embeddings_from_file
from connectors.vector_db_connector import MilvusConnector


def upload_collection_from_file(file_path,
                                collection_name,
                                create_index=False,
                                file_type='pdf',
                                vector_field_name='vector_field',
                                dim=384):
    """
    A function used to upload collection from file. First creates, embeddings, then uploads using the default
    schema for a collection.
    :param file_path: The path of the file to be uploaded as collection.
    :param collection_name: The name of the collection.
    :param create_index: Bool, create index if True.
    :param file_type: The type of file that is being read from the file_path.
    :param vector_field_name: The name of the field that you want to vectorize and store.
    :param dim: The dimension size of the embeddings/vector field.
    """
    # first check if collection exists.
    connector = MilvusConnector()
    connector.connect_to_db()
    connector.set_collection_name(collection_name)
    collection_exists = connector.check_collection_exists()  # this returns a boolean.

    # if the collection exists, we will proceed to do the rest. Otherwise, we will not.
    if not collection_exists:
        # we will proceed to create the embeddings.
        print("Creating Embeddings of the File.")
        vector_field = embeddings_from_file(file_path=file_path,
                                            file_type=file_type)
        # both of the variables provided below will need to move as config parameter values to be set.
        field_name = vector_field_name
        dim = dim

        # We will use the default schema to upload the collection.
        connector.create_default_schema(
            primary_id='primary_key',
            field_name=field_name,
            embeddings_dim=dim
        )
        connector.create_collection()
        print("Collection Created.")

        # fetch the created schema and collection.
        collection = connector.collection
        entities = [vector_field]
        print("Inserting Collection to Database...")
        t0 = time.time()
        collection.insert(entities)
        ins_rt = time.time() - t0
        print(f"Succeeded inserting in {round(ins_rt, 4)} seconds!")

        print("Flushing...")
        start_flush = time.time()
        collection.flush()
        end_flush = time.time() - start_flush
        print(f"Succeeded in {round(end_flush), 4} seconds!")
        # we will create index if create_index==true.
        if create_index:
            print("Creating Collection Index...")
            connector.create_index(field_name=vector_field)
            print("Finished building Index.")

        connector.disconnect()
    else:
        print("""Collection with the provided collection_name already exists.
        Please drop it before uploading again.""")
        connector.disconnect()


def get_milvus_results(collection_name, query, vector_field_name, use_default_schema=True):
    # first connect to milvus to the relevant collection.
    connector = MilvusConnector()
    connector.connect_to_db()
    connector.set_collection_name(collection_name)
    if use_default_schema:
        connector.create_default_schema(
            primary_id='primary_key',
            field_name='vector_field',
            embeddings_dim=384
        )
        connector.create_collection()

    # Create embeddings from a query.
    print("Creating embedding of query.")
    if isinstance(query, str):
        query = [query]
    query_vector = create_embedding(query)
    print("Completed creating embeddings for query.")
    breakpoint()
    print(f"Searching with query vector")
    results = connector.search_collection(
        field_name=vector_field_name,
        query_vector=query_vector
    )
    print(f"Results retrieved.")
    return results


""" To test if uploading collections is successful. """
test_query = ["What is the duration of Maternity Leave"]
query_embedding = create_embedding(test_query)
collection_name = 'eu_maternity'
field = 'vector_field'
result = get_milvus_results(collection_name=collection_name,
                            query=test_query,
                            vector_field_name=field)
breakpoint()
