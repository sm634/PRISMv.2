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
from connectors.elasticsearch_connector import WatsonDiscoveryV2Connector
import json

with open('queries/maternity_cover_queries.json', 'r') as f:
    maternity_json = f.read()
    maternity_json = json.loads(maternity_json)

# organize the collections.
collections = maternity_json['collections']
collections_names = list(collections.keys())

queries = maternity_json['queries']

# instantiate discovery connector instance per collection.
discovery_instances = {}
for collection in collections_names:
    discovery_instances[collection] = WatsonDiscoveryV2Connector()

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

# Get json of query responses
query_responses = json.dumps(all_responses)

with open('data/output/query_responses.json', 'w') as file:
    json.dump(query_responses, file)


def concat_strings(string1, string2, concat_str='-'):
    return string1 + concat_str + string2


def parse_passage_texts(passage_list):
    """
    A helper function that takes in a standard passage list extracted from the response. Parses this list based on
    the number of passages set in the config to be retrieved per query and returns the individual passage text,
    without associated metadata in a list.
    """
    return [passage_list[0][i]['passage_text'] for i in range(0, len(passage_list))]


def format_response_output(response_json, most_relevant_passage: bool = True):
    """
    A function that breaks down a hierarchy of the query response output to a format that can be passed to a LLM.
    :param response_json: Dict with the results taken from the discovery response.
    :param most_relevant_passage: Bool type which indicates of all the passage texts returned per query, whether to
    only keep the most relevant (highest ranked) one.
    """
    collection_names = list(response_json.keys())

    output = {}
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
        output[collection] = collection_output

    return output


output_dict = format_response_output(response_json=all_responses, most_relevant_passage=True)

output_response = json.dumps(output_dict)
with open('data/output/query_passage_formatted.json', 'w') as file:
    json.dump(output_response, file)
