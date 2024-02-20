from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from utils.timestamps import get_stamp
from utils.files_handler import FileHandler
from utils.models_funcs import get_model
import json

# global variable to process required files for data and prompts.
file_handler = FileHandler()


def prompt_inputs(topic, input_text):
    """
    Temporary function for article classifier which takes one argument, which is to be mapped ot the input data.
    :param topic: The topic/name of the argument to be passed into the prompt template.
    :param input_text: The input/text that is passed as an article.
    :return: A dictionary that can be passed to a Langchain run command.
    """
    return {topic: input_text}


with open('data/queries/output/query_passage_formatted.json', 'r') as f:
    file_content = f.read()
    file_json = json.loads(file_content)

if isinstance(file_json, str):
    file_json = json.loads(file_json)

query = "What is the duration of Maternity Leave"
eu_example = file_json['eu-maternity'][query]
uk_example = file_json['uk-maternity'][query]

print(eu_example + "\n\n")
print(uk_example)

breakpoint()
