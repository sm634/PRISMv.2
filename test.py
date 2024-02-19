from connectors.elasticsearch_connector import WatsonDiscoveryV2Connector

watson_connector_dora = WatsonDiscoveryV2Connector()
watson_connector_cra = WatsonDiscoveryV2Connector()

query = 'Proportionality Principle'
print("The query is: ", query)

collection_ids = {
    "DORA": ["ae5d8b24-7c12-8750-0000-018da2dbd6aa"],
    "CRA": ["1ca3aab4-6ada-8228-0000-018da2dc9b5a"]
}

# set collection
collection1 = "DORA"

print(f"sending query for {collection1} collection...")
watson_connector_dora.query_response(
    query=query,
    collection_ids=collection_ids["DORA"]
)
print("query response received for DORA collection")

DORA_document_passages = watson_connector_dora.get_document_passages()
DORA_subtitles = watson_connector_dora.get_subtitle()
DORA_text = watson_connector_dora.get_text()
DORA_result_metadata = watson_connector_dora.get_result_metadata()

# set collection
collection2 = "CRA"

print(f"sending query for {collection2} collection...")
watson_connector_cra.query_response(
    query=query,
    collection_ids=collection_ids["CRA"]
)
print("query response received for DORA collection")

CRA_document_passages = watson_connector_cra.get_document_passages()
CRA_subtitles = watson_connector_cra.get_subtitle()
CRA_text = watson_connector_cra.get_text()
CRA_result_metadata = watson_connector_cra.get_result_metadata()

breakpoint()
