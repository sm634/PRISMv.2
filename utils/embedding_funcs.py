"""
Docstring
---------
Functions to create embeddings. Needs to be updated to have more models incorporated and also part of the BaseModel
and Model Connector classes (with config for models). Currently only uses the SentenceTransformer class.
"""
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from typing import List
import re

models = [
    'all-MiniLM-L6-v2'
]

FILE_TYPE_TXT = "txt"
FILE_TYPE_PDF = "pdf"


def get_collection_from_file(file_path, file_type):
    """
    A function that takes in a file from local and creates a collection from it.
    """
    collection = None
    # load the relevant file based on file type (currently supports pdfs and txt).
    if file_type == FILE_TYPE_TXT:
        loader = TextLoader(file_path, encoding="1252")
        collection = loader.load()
    elif file_type == FILE_TYPE_PDF:
        loader = PyPDFLoader(file_path)
        collection = loader.load()

    return collection


def create_documents(collection: list, chunk_size=500, chunk_overlap=50, custom_splitter=True, splitter='recursive'):
    """
    A function that takes in a whole file collection and chunks the text into separate documents.
    """
    if custom_splitter:
        if str(type(collection[0])) == "<class 'langchain_core.documents.base.Document'>":
            collection_content = ""
            for doc in collection:
                collection_content += doc.page_content
            documents_content = re.split(r'\n \n\d.|\n\d.', collection_content)

        else:
            documents_content = [re.split(r'\n \n\d.|\n\d.', doc) for doc in collection]
        doc_metadata = [doc.metadata for doc in collection]

    else:
        text_splitter = None
        print("""Creating documents from text collection.""")
        if splitter == 'character':
            text_splitter = CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                is_separator_regex=False
            )
        elif splitter == 'recursive':
            text_splitter = RecursiveCharacterTextSplitter(
                separators=[r'\n \n\d.'],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                is_separator_regex=True
            )
        documents = text_splitter.split_documents(collection)
        documents_content = [doc.page_content for doc in documents]
        doc_metadata = [doc.metadata for doc in documents]

    print("Completed chunking and creating documents and metadata.")
    return documents_content, doc_metadata


def create_embedding(documents: List, embedding_model='all-MiniLM-L6-v2'):
    # instantiate the sentence transformer embeddings model.
    model = SentenceTransformer(embedding_model)
    print(f"Instantiated Sentence Transformer Model: {model}. Creating Embeddings.")

    embeddings = []
    for document in documents:
        doc_vector = model.encode(document)
        embeddings.append(doc_vector)
    print("Finished creating embeddings.")
    return embeddings


def embeddings_from_file(file_path, file_type, return_dict=False):
    collection = get_collection_from_file(
        file_path=file_path,
        file_type=file_type
    )
    documents, metadata = create_documents(collection=collection)
    embeddings = create_embedding(documents)

    if return_dict:
        assert len(documents) == len(embeddings)
        output_dict = {}
        for i in range(0, len(embeddings)):
            doc = documents[i]
            vector = embeddings[i]
            output_dict[doc] = vector
        return output_dict

    else:
        return embeddings
