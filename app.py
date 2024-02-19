import streamlit as st
import subprocess
from utils.files_handler import FileHandler

file_handler = FileHandler()
file_handler.get_config()
config = file_handler.config

# Define configuration options
config_options = {
    "MODEL_PROVIDERS": ['OPENAI', 'WATSONX'],
    "TASK": ["ARTICLE_CLASSIFIER", "EMBEDDINGS_COMPARISON", "PREPROCESS_ARTICLE"]
}

# Streamlit UI components
st.title("Configuration Selector")

# the config needs to be updated based on what is selected in the UI here.
model_provider = st.selectbox("MODEL PROVIDER", config_options['MODEL_PROVIDERS'])
task = st.selectbox("TASK", config_options["TASK"])


# Function to run Python script with selected option
def run_script():
    script_path = "main.py"  # Replace with the path to your Python script
    subprocess.run(["python", script_path])


# Button to run the script
if st.button("Run Script"):
    config["MODEL_PROVIDER"] = model_provider
    config["TASK"] = task
    st.write("UPDATING CONFIG")
    file_handler.write_config(config)
    st.write("RUNNING SCRIPT")
    run_script()
    st.write("FINISHED TASK")
