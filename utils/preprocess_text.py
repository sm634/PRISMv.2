import re
import string
from utils.files_handler import FileHandler
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from utils.models_funcs import get_model

# instantiate file handler.
file_handler = FileHandler()


class StandardTextCleaner:

    @staticmethod
    def remove_special_characters(text):
        # Remove special characters
        text = re.sub(r'[^a-zA-Z0-9\s@.]', '', text)

        # Remove non-English characters
        text = ''.join(char for char in text if char in string.ascii_letters or char.isspace() or char in "@.")

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        return text


class TextDenoiser:
    def __init__(self):
        self.denoiser_model = get_model()

    @staticmethod
    def __prompt_inputs(key, input_text):
        """
        Temporary function for article classifier which takes one argument, which is to be mapped ot the input data.
        :param topic: The topic/name of the argument to be passed into the prompt template.
        :param input_text: The input/text that is passed as an article.
        :return: A dictionary that can be passed to a Langchain run command.
        """
        return {key: input_text}

    def remove_adverts(self, input_text, prompt_template_file, llm_model):
        """
        A denoiser that uses LLMs to remove advert from article text that has been scraped from a webpage.
        :param llm_model: the llm to be used for removing noise from raw data.
        :param input_text: The text for which the noise is to be removed.
        :param prompt_template_file: The file name of the prompt template to be used to get the LLM to remove the noise.
        :return: Str, that is the same as the input_text but with the noise removed.
        """
        # set up the prompt template
        ad_remover_template = file_handler.get_prompt_template(file_name=prompt_template_file)
        prompt_template = PromptTemplate.from_template(ad_remover_template)

        llm_chain = LLMChain(prompt=prompt_template, llm=llm_model)

        output = llm_chain.run(
            self.__prompt_inputs('raw_data', input_text)
        )
        return output
