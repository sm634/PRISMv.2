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
from utils.discovery_response_handler import get_discovery_data
from utils.models_funcs import get_model
from utils.files_handler import FileHandler
from utils.preprocess_text import StandardTextCleaner

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import pandas as pd

import os

file_handler = FileHandler()


def prompt_inputs(topic1, passage1, topic2, passage2):
    """
    Temporary function for the policy comparator which takes two arguments, which is mapped to the input data.
    :param topic1: The topic/name of the argument to be passed into the prompt template.
    :param passage1: The passage that is passed as the first comparator.
    :param topic2: The topic/name of the second argument to be passed into the prompt template.
    :param passage2: The 2nd passage that is passed as the 2nd comparator.
    :return: A dictionary that can be passed to a Langchain run command.
    """
    return {topic1: passage1, topic2: passage2}


def prep_passages_for_llms(formatted_json, clean_passages=True):
    """
    A helper function that prepares the data to be passed into the LLM for comparison.
    :param formatted_json: The input will be the formatted response extracted from the search engine. The hierarchy
    assumes Dict(collection: {query: passage}) for m-collections and n-query:passage(s) pairs.
    - collection
        - query: passage
    :param clean_passages: Bool if true, uses standard text cleaner to remove special characters from the text.
    :return: Dict(List) of passages from the different collections and queries to be compared,
    such as query: [passage1_collection1, passage1_collection2].
    """
    # parse the expected dict structure to recombine and return the new output structure.
    collection_names = list(formatted_json.keys())
    collection_sample = collection_names[0]
    queries = list(formatted_json[collection_sample].keys())

    # instantiate the standard text cleaner.
    standard_text_cleaner = StandardTextCleaner()

    output = {}
    for collection in collection_names:
        query_passages_dict = formatted_json[collection]
        for i, query in enumerate(queries):
            # first clean each passage if that option is on.
            if clean_passages:
                passage = standard_text_cleaner.remove_special_characters(query_passages_dict[query])
            else:
                passage = query_passages_dict[query]

            # Start storing the passage.
            if query not in output.keys():
                # Store the passage in a list that is a value to the key, which is the query.
                output[query] = [passage]
            else:
                # If that query already has an entry in the dictionary, then append the passage to the list of passages
                # related to that query.
                output[query].append(passage)

    return output


def run_text_comparator(use_existing_outputs=True):
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

        """First LLM analysis."""
        # prepare the prompt template.
        prompt_file = file_handler.get_prompt_template(file_name='compare_text.txt')
        prompt_template = PromptTemplate.from_template(prompt_file)

        # instantiate model with prompt.
        llm_chain = LLMChain(prompt=prompt_template, llm=model)

        # run the llm on all passages.
        """THIS IS TEMPORARY CODE THAT NEEDS TO BE GENERALIZABLE FOR MORE THAN TWO PASSAGES"""
        llm_analyses = []
        passages_collection1 = []
        passage_collection2 = []
        queries = list(llm_data.keys())

        print("Invoking LLM for Analysis")
        for query in queries:
            # prepare the passages by extracting it from the dict(list).
            passage_1 = llm_data[query][0]
            passage_2 = llm_data[query][1]
            passages_collection1.append(passage_1)
            passage_collection2.append(passage_2)

            llm_analysis = llm_chain.invoke(
                prompt_inputs('passage1', passage_1, 'passage2', passage_2)
            )
            # As the llm_chain.invoke function returns a dict with the input variables and the text returned by llm,
            # we will only keep the text that is returned.
            llm_analyses.append(llm_analysis['text'])

        print("LLM Analysis complete")

        """Second LLM analysis."""
        # Get the prompt template for the second invocation of the llm.
        prompt_file2 = file_handler.get_prompt_template(file_name='generate_policy_guidance.txt')
        prompt_template2 = PromptTemplate.from_template(prompt_file2)

        # instantiate model with prompt.
        llm_chain = LLMChain(prompt=prompt_template2, llm=model)

        # run the llm for second layer of processing on all passages.
        llm_generation = []
        print("Invoking LLM for Generation")
        for i in range(0, len(queries)):
            llm_analysis = llm_chain.invoke(
                prompt_inputs('VERSION_1', passages_collection1[i], 'VERSION_2', passage_collection2[i])
            )
            # As the llm_chain.invoke function returns a dict with the input variables and the text returned by llm,
            # we will only keep the text that is returned.
            llm_generation.append(llm_analysis['text'])

        df = pd.DataFrame(
            {
                           'query': queries,
                           'eu_maternity_passage': passages_collection1,
                           'uk_maternity_passage': passage_collection2,
                           'llm_analysis': llm_analyses,
                           'generated_guidance': llm_generation
            }
        )

        output_file_name = f'text_comparator_{model_name}'
        file_handler.save_df_to_csv(df=df, file_name=output_file_name)
