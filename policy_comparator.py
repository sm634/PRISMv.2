"""
Docstring
---------

This script takes in a RAG based solution to compare documents (selected to be most relevant to one another) from
different collections. Each collection may represent a Policy such as on Maternity Leave across the UK vs. the EU
for example.

The execution is as follows:
1. Define a set of queries to pass for search (for example, 'what is the maternity pay')
2. Retrieve the relevant chunks or documents have been retrieved from from the separate collections.
3. Compare those chunks to one another, per query and identify similarities and differences using a suitable LLM.
"""
from utils.discovery_response_handler import extract_query_from_json, get_discovery_data
from utils.models_funcs import get_model
from utils.files_handler import FileHandler

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

import os

file_handler = FileHandler()


def prep_passages_for_llms(formatted_json):
    """
    A helper function that prepares the data to be passed into the LLM for comparison.
    :param formatted_json: The input will be the formatted response extracted from the search engine. The hierarchy
    assumes Dict(collection: {query: passage}) for m-collections and n-query:passage(s) pairs.
    - collection
        - query: passage
    :return: Dict(List) of passages from the different collections and queries to be compared,
    such as query: [passage1_collection1, passage1_collection2].
    """
    # parse the expected dict structure to recombine and return the new output structure.
    collection_names = list(formatted_json.keys())
    collection_sample = collection_names[0]
    queries = list(formatted_json[collection_sample].keys())

    output = {}
    for collection in collection_names:
        query_passages_dict = formatted_json[collection]
        for i, query in enumerate(queries):
            if query not in output.keys():
                # Store the passage in a list that is a value to the key, which is the query.
                output[query] = [query_passages_dict[query]]
            else:
                # If that query already has an entry in the dictionary, then append the passage to the list of passages
                # related to that query.
                output[query].append(query_passages_dict[query])

    return output


def run_policy_comparator(use_existing_outputs=True):
    """
    A function to run the policy comparator.
    """
    file_handler.get_config(config_file_name='search_selector')
    search_engine = file_handler.config['SEARCH_ENGINE']

    if search_engine.lower() == 'discovery':
        if use_existing_outputs:

            import json

            folder_path = 'data/queries/output/'
            output_files_list = os.listdir(folder_path)
            file_path = folder_path + output_files_list[0]

            with open(file_path, 'r') as f:
                file_content = f.read()
                discovery_output = json.loads(file_content)

            if isinstance(discovery_output, str):
                discovery_output = json.loads(discovery_output)

        else:
            discovery_output = get_discovery_data(
                queries_json_input='maternity_cover_queries.json',
                save_output=True
            )

        # get the model
        model_dict = get_model()
        model = model_dict['model']
        model_name = model_dict['name']

        # get the collections and queries.
        llm_data = prep_passages_for_llms(discovery_output)

        
