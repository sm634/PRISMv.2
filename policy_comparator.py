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


with open('data/output/query_responses.json', 'w') as file:
    try:
        json.dump(all_responses, file)
    except:
        output_response = json.dumps(all_responses)
        json.dump(output_response, file)
