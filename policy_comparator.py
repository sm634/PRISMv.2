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
import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

file_handler = FileHandler()


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
        collections = list(discovery_output.keys())
        collection_sample = collections[0]
        queries = list(discovery_output[collection_sample].keys())
        

