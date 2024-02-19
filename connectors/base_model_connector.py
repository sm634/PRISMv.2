"""
Docstring
---------

The base class to connect to models. There are currently supported for LLM foundation models from OpenAI and Watsonx.
The particular values assigned to parameters specified within this class will depend on the config file. Once the
values have been assigned to the relevant class attributes, these will be inherited in ModelsConnector class.
"""
import os

from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

from dotenv import load_dotenv
from utils.files_handler import FileHandler


class BaseModelConnector:

    def __init__(self):
        """
        BaseModelConnector reads in the models_config.yaml file and sets up the relevant variable values to instantiate the
        required model to be initialized using hte ModelConnector class, handling dependencies for model access and
        inference.
        """
        # we will need the environment variables for credentials.
        global provider_task
        load_dotenv()

        # instantiate the credentials values
        self.api_key = ''
        self.project_id = ''
        self.model_endpoint = ''

        # get models configs
        file_handler = FileHandler()
        file_handler.get_config()
        self.config = file_handler.config

        # get the model provider and the task of interest.
        self.model_provider = self.config['MODEL_PROVIDER'].lower()
        self.task = self.config['TASK'].lower()

        if self.model_provider == 'openai':
            # get the credentials
            self.api_key = os.environ['OPENAI_API_KEY']

            if self.task == 'article_classifier':
                provider_task = self.config['OPENAI']['ARTICLE_CLASSIFIER']
            elif self.task == 'preprocess_article':
                provider_task = self.config['OPENAI']['PREPROCESS_ARTICLE']

        elif self.model_provider == 'watsonx':
            # get the watsonx credentials
            self.api_key = os.environ['WATSONX_API_KEY']
            self.project_id = os.environ['PROJECT_ID']
            self.model_endpoint = os.environ['MODEL_ENDPOINT']

            if self.task == 'article_classifier':
                provider_task = self.config['WATSONX']['ARTICLE_CLASSIFIER']
            elif self.task == 'preprocess_article':
                provider_task = self.config['WATSONX']['PREPROCESS_ARTICLE']
        else:
            raise

        model_type = provider_task['model_type']
        if self.model_provider == 'watsonx':
            self.model_type = getattr(ModelTypes, model_type)
            self.model_name = self.model_type.name
        else:
            self.model_type = model_type
            self.model_name = model_type

        # decoding method
        decoding_method = provider_task['decoding_method']
        if self.model_provider == 'watsonx':
            self.decoding_method = getattr(DecodingMethods, decoding_method)
        else:
            self.decoding_method = decoding_method

        # set the hyperparameters according to the values in the config file.
        self.max_tokens = provider_task['max_tokens']
        self.min_tokens = provider_task['min_tokens']
        self.temperature = provider_task['temperature']
        self.top_p = provider_task['top_p']
        self.top_k = provider_task['top_k']
        self.repetition_penalty = provider_task['repetition_penalty']

        self.params = {
            GenParams.MAX_NEW_TOKENS: self.max_tokens,
            GenParams.MIN_NEW_TOKENS: self.min_tokens,
            GenParams.DECODING_METHOD: self.decoding_method,
            GenParams.TEMPERATURE: self.temperature,
            GenParams.TOP_P: self.top_p,
            GenParams.TOP_K: self.top_k,
            GenParams.REPETITION_PENALTY: self.repetition_penalty
        }
