import streamlit as st
from utils.files_handler import FileHandler
from src.preprocess_pipeline import StandardTextCleaner

from utils.discovery_response_handler import parse_passage_texts
from connectors.elasticsearch_connector import WatsonDiscoveryV2Connector

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from utils.models_funcs import get_model

file_handler = FileHandler()
file_handler.get_config()
config = file_handler.config
standard_text_cleaner = StandardTextCleaner()

# Get collections and queries json.
file_handler.get_queries_from_json('maternity_cover_queries.json')
queries_json: dict = file_handler.queries_json

# store the metadata variables, which can be used to display certain answers.
queries = queries_json['queries']
collections_names = list(queries_json['collections'])

# Define configuration options
config_options = {
    "Queries": queries,
    "Collections": collections_names
}

# Streamlit UI components
st.title("Demo Policy Comparator")

# create a subheading
st.subheader("Collections to Use")

# Create a container for the layout
col1, col2 = st.columns([1, 1])

# Text on the left
with col1:
    collection_1 = st.selectbox("Collection 1", config_options['Collections'])

# Text on the right
with col2:
    collection_2 = st.selectbox("Collection 2", config_options['Collections'])

st.subheader("Query to Run")
# the config needs to be updated based on what is selected in the UI here.
queries_selector = st.selectbox("Queries", config_options['Queries'])


def get_most_relevant_discovery_passage(query, collection_id):
    """A function that takes in the selected query and runs that to return the relevant passages from Discovery"""
    discovery_connector = WatsonDiscoveryV2Connector()
    discovery_connector.query_response(
        query=query,
        collection_ids=[collection_id]
    )
    passages_list = discovery_connector.get_document_passages()
    most_relevant_passage = parse_passage_texts(passages_list)[0]

    return most_relevant_passage


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


# Function to run Python script with selected option
def get_comparison(passage_1, passage_2):
    # get the model
    model_dict = get_model()
    model = model_dict['model']
    model_name = model_dict['name']

    # prepare the prompt template.
    prompt_file = file_handler.get_prompt_template(file_name='compare_text.txt')
    prompt_template = PromptTemplate.from_template(prompt_file)

    # instantiate model with prompt.
    llm_chain = LLMChain(prompt=prompt_template, llm=model)
    llm_analysis = llm_chain.invoke(
        prompt_inputs('passage1', passage_1, 'passage2', passage_2)
    )

    return {'model_name':model_name,
            'llm_analysis': llm_analysis['text']}


# Button to run the script
if st.button("Get Comparison"):
    st.subheader("Retrieving Documents")
    # set the selected values to use.
    collection_id1 = queries_json['collections'][collection_1]
    collection_id2 = queries_json['collections'][collection_2]
    query = queries_selector

    # Retrieve and show Passage 1.
    markdown1 = f"**PASSAGE 1 {collection_1} SEARCH**"
    st.markdown(markdown1)
    # show the first passage returned.
    passage1 = get_most_relevant_discovery_passage(query=query, collection_id=collection_id1)
    passage1 = standard_text_cleaner.remove_special_characters(passage1)
    st.write(passage1)

    # Retrieve and show Passage 2.
    markdown2 = f"**PASSAGE FROM {collection_2} SEARCH**"
    st.markdown(markdown2)
    # show the first passage returned.
    passage2 = get_most_relevant_discovery_passage(query=query, collection_id=collection_id2)
    passage2 = standard_text_cleaner.remove_special_characters(passage2)
    st.write(passage2)

    # Generate Comparison From the LLM.
    st.subheader("LLM Analysis")
    output_dict = get_comparison(passage_1=passage1, passage_2=passage2)
    # parse the output to display.
    model_used = output_dict['model_name']
    llm_response = output_dict['llm_analysis']
    # display llm and it's response.
    markdown3 = f"**Analysis from {model_used}**"
    st.markdown(markdown3)
    st.write(llm_response)

    st.subheader("Finished Analysis. Would you like to try another?")