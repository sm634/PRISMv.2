"""
Docstring
---------

A script that will store certain functions related to calling/instantiating LLM models and reading the relevant
templates from that LLM to be used for the chosen task.
"""
from connectors.models_connector import ModelConnector


def get_model():
    """
    A function to instantiate the particular instance of the model desired.
    :return: Foundation model of choice.
    """

    model_client = ModelConnector()
    # get a particular instance of the model of choice
    model = model_client.instantiate_model()
    model_name = model_client.model_name

    return {'name': model_name,
            'model': model}
