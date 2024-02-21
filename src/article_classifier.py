from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from utils.files_handler import FileHandler
from utils.models_funcs import get_model

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


def run_article_classifier():
    """
    Run the entire pipeline E2E.
    """
    # get the model
    model_dict = get_model()
    model = model_dict['model']
    model_name = model_dict['name']

    # get the prompt template
    red_flag_template = file_handler.get_prompt_template(file_name='classify_article.txt')
    prompt_template = PromptTemplate.from_template(red_flag_template)

    # get the data
    df = file_handler.get_df_from_file(file_name='First200_ic.csv')
    sample_articles = df[['_id',
                          'article',
                          'classification.isIncident']].loc[df['classification.isIncident'] == 'Incident']

    sample_articles = sample_articles.sample(10)

    # instantiate model
    llm_chain = LLMChain(prompt=prompt_template, llm=model)

    # new col name
    new_col = model_name + '_classification'
    # apply the model on the sample articles and store in a new column.
    sample_articles[new_col] = sample_articles['article'].apply(lambda x:
                                                                llm_chain.invoke(
                                                                    prompt_inputs('article', x)
                                                                )
                                                                )

    """OUTPUT"""
    # standardize the output format.
    sample_articles.set_index('_id', inplace=True)

    # define the output name.
    output_name = f'sample_classification_{model_name}.csv'

    # save the new output to data outputs.
    file_handler.save_df_to_csv(df=sample_articles, file_name=output_name)
