import streamlit as st
import subprocess
from utils.files_handler import FileHandler

# get models config.
file_handler = FileHandler()
file_handler.get_config()
config = file_handler.config

# get arguments config
file_handler.get_config('arguments_passer.yaml')
arguments_config = file_handler.config

# Define configuration options
config_options = {
    "MODEL_PROVIDERS": ['WATSONX', 'OPENAI'],
    "TASK": ["TEXT_COMPARATOR",
             "EMBEDDINGS_COMPARATOR",
             "ARTICLE_CLASSIFIER",
             "REDFLAG_ARTICLE_COMPARATOR",
             "PREPROCESS_ARTICLE",
             ]
}
models_config = {
    'OPENAI':
        {
            'model': ['gpt-3.5-turbo-0301']
        },
    'WATSONX':
        {
            'model': ['LLAMA_2_70B_CHAT',
                      'LLAMA_2_13B_CHAT',
                      'GRANITE_13B_CHAT_V2',
                      'GRANITE_13B_CHAT_INSTRUCT_V2',
                      'FLAN_UL2',
                      'MTP_7B_INSTRUCT2']
        }
}

# Streamlit UI components
st.title("Configuration Selector")

# the config needs to be updated based on what is selected in the UI here.
model_provider = st.selectbox("MODEL PROVIDER", config_options['MODEL_PROVIDERS'])
task = st.selectbox("TASK", config_options["TASK"])
if model_provider == 'OPENAI':
    model = st.selectbox("Model", models_config["OPENAI"]["model"])
elif model_provider == 'WATSONX':
    model = st.selectbox("Model", models_config["WATSONX"]["model"])
else:
    model_provider = "WATSONX"
    model = "LLAMA_2_70B_CHAT"

col1, col2 = st.columns([1, 1])
# options for Embeddings Comparison.
if task == 'EMBEDDINGS_COMPARATOR':
    invoke_llm_options = {
        "LLM Analysis": [True, False],
        "LLM Draft Policy Generation": [False, True]
    }
    with col1:
        invoke_llm_analysis = st.selectbox("Invoke Embeddings Comparator LLM Analysis",
                                           invoke_llm_options['LLM Analysis'])
    with col2:
        if invoke_llm_analysis:
            invoke_llm_generation = st.selectbox("Generate Embeddings Comparator Draft Policy",
                                                 invoke_llm_options['LLM Draft Policy Generation'])
        else:
            invoke_llm_generation = False

col3, col4 = st.columns([1, 1])
# options for Text Comparison.
if task == 'TEXT_COMPARATOR':
    invoke_llm_options = {
        "LLM Analysis": [True, False],
        "LLM Draft Policy Generation": [False, True]
    }
    with col3:
        invoke_llm_analysis = st.selectbox("Invoke Text Comparator LLM Analysis",
                                           invoke_llm_options['LLM Analysis'])
    with col4:
        if invoke_llm_analysis:
            invoke_llm_generation = st.selectbox("Generate Text Comparator Draft Policy",
                                                 invoke_llm_options['LLM Draft Policy Generation'])
        else:
            invoke_llm_generation = False


# Function to run Python script with selected option
def run_script():
    script_path = "main.py"  # Replace with the path to your Python script
    subprocess.run(["python", script_path])


# Button to run the script
if st.button("Run Script"):
    config["MODEL_PROVIDER"] = model_provider
    config["TASK"] = task
    config[model_provider][task]["model_type"] = model
    # save arguments.
    if task == 'EMBEDDINGS_COMPARATOR':
        arguments_config['EMBEDDINGS_COMPARATOR']['INVOKE_LLM_ANALYSIS'] = invoke_llm_analysis
        arguments_config['EMBEDDINGS_COMPARATOR']['INVOKE_LLM_GENERATION'] = invoke_llm_generation
        file_handler.write_config(arguments_config, config_file_name='arguments_passer.yaml')
    if task == 'TEXT_COMPARATOR':
        arguments_config['TEXT_COMPARATOR']['INVOKE_LLM_ANALYSIS'] = invoke_llm_analysis
        arguments_config['TEXT_COMPARATOR']['INVOKE_LLM_GENERATION'] = invoke_llm_generation
        file_handler.write_config(arguments_config, config_file_name='arguments_passer.yaml')
    st.subheader("UPDATING CONFIG")
    file_handler.write_config(config)
    st.subheader("RUNNING SCRIPT...")
    run_script()
    st.subheader("FINISHED TASK")
