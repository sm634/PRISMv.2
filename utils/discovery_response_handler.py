"""
Docstring
---------

This is a set of helper functions to parse responses from elasticsearch discovery v2 in IBM Watson.
"""
from connectors.elasticsearch_connector import WatsonDiscoveryV2Connector
from utils.files_handler import FileHandler

file_handler = FileHandler()


def extract_query_from_json(queries_json_input: str):
    """
    A function that parses the expected json structure from the input queries and returns a tuple of each.
    :param queries_json_input: The json file name with queries and collections metadata to be read and parsed.
    return: Tuple of collections: List, collections_names: List, queries: List
    """
    file_handler.get_queries_from_json(queries_json_input)
    queries_json: dict = file_handler.queries_json
    # organize the collections.
    collections: dict = queries_json['collections']
    collections_names: list = list(collections.keys())
    # get the queries
    queries: list = queries_json['queries']

    return collections, collections_names, queries


def instantiate_collections_discovery_instances(queries_json_input: str, return_all=True):
    """
    Provide the desired collections for which a Discovery instance is to be instantiated.
    :param queries_json_input: The json file name with queries and collections metadata to be read and parsed.
    :param return_all bool: An option to return all the extracted collections metadata, queries and the instantiated
    discovery instances.
    return: Dict of collection: discovery_instance object for each collection.
    """
    collections, collections_names, queries = extract_query_from_json(queries_json_input)
    # instantiate discovery connector instance per collection.
    discovery_instances = {}
    for collection in collections_names:
        discovery_instances[collection] = WatsonDiscoveryV2Connector()
    # the return all option will provide all the data extracted from here.
    if return_all:
        output = collections, collections_names, queries, discovery_instances
        return output
    else:
        return discovery_instances


def get_discovery_responses(queries_json_input: str, save_output: bool = True):
    """
    A function that is used to connect to the discovery instances per collection with the required queries and
    extracting some information from that to store in a new dict/json format.
    :param queries_json_input: The json file name with queries and collections metadata to be read and parsed.
    :param save_output bool: If true, save the output to a json file in data/output.
    return Dict: Discovery response json.
    """
    # instantiate the discovery instances.
    collections, collections_names, queries, discovery_instances = instantiate_collections_discovery_instances(
        queries_json_input=queries_json_input,
        return_all=True
    )
    # collect all responses on a dictionary by collection.
    all_responses = {}
    for collection in list(discovery_instances.keys()):
        discovery_connector = discovery_instances[collection]
        # create a list to store responses for a collection for all queries.
        query_responses = all_responses[collection] = []
        for query in queries:
            print(f"Running query: '{query}' against {collection}")
            # send the query to retrieve response from discovery.
            discovery_connector.query_response(
                query=query,
                collection_ids=[collections[collection]]
            )
            # store selected response data.
            query_responses.append(
                {
                    'query': query,
                    'passages': discovery_connector.get_document_passages(),
                    'subtitles': discovery_connector.get_subtitle(),
                    'text': discovery_connector.get_text(),
                    'result_metadata': discovery_connector.get_result_metadata()
                }
            )

    # needs to be reimplemented with file handler, including timestamp.
    if save_output:
        file_handler.save_to_json(
            all_responses,
            file_name='query_responses'
        )

    return all_responses


def concat_strings(string1, string2, concat_str='-'):
    return string1 + concat_str + string2


def parse_passage_texts(passage_list):
    """
    A helper function that takes in a standard passage list extracted from the response. Parses this list based on
    the number of passages set in the config to be retrieved per query and returns the individual passage text,
    without associated metadata in a list.
    """
    return [passage_list[0][i]['passage_text'] for i in range(0, len(passage_list))]


def format_response_output(
        response_json,
        most_relevant_passage: bool = True,
        save_output: bool = True
):
    """
    A function that breaks down a hierarchy of the query response output to a format that can be passed to a LLM.
    :param response_json: Dict with the results taken from the discovery response.
    :param most_relevant_passage: Bool type which indicates of all the passage texts returned per query, whether to
    only keep the most relevant (highest ranked) one.
    :param save_output: If true, save the output as a json file.
    return: Dict containing formatted output from the Discovery API response.
    """
    collection_names = list(response_json.keys())

    formatted_response_output = {}
    for collection in collection_names:
        # get collection data which is a list of dictionaries, where we have a dictionary per query.
        collection_data: list = response_json[collection]
        # we will get the requisite data associated per query as n. of queries determines n. of data points.
        collection_output = {}
        for _, packet in enumerate(collection_data):
            # The json 'packet' has query, passages, subtitles, text and result metadata as keys.
            query = packet['query']
            # get only the most relevant passage from the set of retrieved documents.
            passages_list = packet['passages']
            # get all passage texts without the metadata in a list
            passages_text_list = parse_passage_texts(passage_list=passages_list)
            # if most_relevant_passage, then only take the highest ranked passage text.
            if most_relevant_passage:
                output_passages = passages_text_list[0]
            else:
                output_passages = passages_text_list

            # Simplify output with just the query against the output passage.
            collection_output[query] = output_passages
        # The output is a dictionary with all the query-passage pairs retrieved per collection.
        formatted_response_output[collection] = collection_output

    # needs to be reimplemented with file handler, including timestamp.
    if save_output:
        file_handler.save_to_json(
            data=formatted_response_output,
            file_name='query_passage_formatted'
        )

    return formatted_response_output


def get_discovery_data(
        queries_json_input,
        save_output: bool = True):
    """
    A function that uses previous functions to get the desired data from the discovery responses in a dict, with the
    option to save the output as a json file.
    :param queries_json_input: The file name of the json file containing collections metadata and queries.
    :param save_output: If true, save the output as a json file.
    """
    # get the initial un-formatted discovery responses.
    discovery_responses = get_discovery_responses(
        queries_json_input,
        save_output=save_output
    )

    # get the formatted discovery responses.
    formatted_discovery_responses = format_response_output(
        response_json=discovery_responses,
        most_relevant_passage=True,
        save_output=save_output
    )

    return formatted_discovery_responses
