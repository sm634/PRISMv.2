from connectors.base_model_connector import BaseModelConnector

from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM

from langchain_community.chat_models import ChatOpenAI


class ModelConnector(BaseModelConnector):

    def __init__(self):
        """
        The ModelConnector class initializing the specified model (based on config) and initialises the hyperparameters,
        many can be used/is relevant for different model providers. These are all integrated with LangChain, hence the
        set of applications are highly dependent on Langchain model capabilities.
        """
        # The model provider will be from a list of model providers.
        super().__init__()

        self.params = {
            GenParams.MAX_NEW_TOKENS: self.max_tokens,
            GenParams.MIN_NEW_TOKENS: self.min_tokens,
            GenParams.DECODING_METHOD: self.decoding_method,
            GenParams.TEMPERATURE: self.temperature,
            GenParams.TOP_P: self.top_p,
            GenParams.TOP_K: self.top_k,
            GenParams.REPETITION_PENALTY: self.repetition_penalty
        }

    def instantiate_model(self):

        if self.model_provider == 'watsonx':
            model = Model(
                model_id=self.model_type,
                params=self.params,
                credentials={
                    "apikey": self.api_key,
                    "url": self.model_endpoint
                },
                project_id=self.project_id
            )
            # integrate with langchain Watsonx LLM model
            model = WatsonxLLM(model=model)
            return model

        elif self.model_provider == 'openai':

            model = ChatOpenAI(
                api_key=self.api_key,
                temperature=self.temperature,
                model=self.model_type
            )

            return model

        else:
            pass
