from utils.embedding_funcs import embeddings_from_file
from sentence_transformers import util
from utils.models_funcs import get_model
from utils.files_handler import FileHandler

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import pandas as pd

file_handler = FileHandler()


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


def run_embeddings_comparator(invoke_llm_analysis=False, invoke_llm_generation=False):
    uk_maternity_embeddings = embeddings_from_file(
        file_path='data/input/UK Maternity cover policies.pdf',
        file_type='pdf',
        return_dict=True
    )
    eu_maternity_embeddings = embeddings_from_file(
        file_path='data/input/EU Maternity cover policies.pdf',
        file_type='pdf',
        return_dict=True
    )

    print("Computing Cosine Similarity between articles and indicators")
    # calculate the similarity scores.
    top_matches = {}
    sim_scores = {}
    # calculate the similarity scores for each article and indicator and store the top_3.
    for uk_doc in list(uk_maternity_embeddings.keys()):
        uk_doc_vector = uk_maternity_embeddings[uk_doc]

        for eu_doc in list(eu_maternity_embeddings.keys()):
            eu_doc_vector = eu_maternity_embeddings[eu_doc]
            sim_score = util.pytorch_cos_sim(a=uk_doc_vector, b=eu_doc_vector)
            sim_score = float(sim_score[0][0])
            if uk_doc in sim_scores.keys():
                if sim_score > sim_scores[uk_doc]:
                    sim_scores[uk_doc] = sim_score
                    top_matches[uk_doc] = eu_doc
            else:
                sim_scores[uk_doc] = sim_score
                top_matches[uk_doc] = eu_doc

    print("Sematic Similarity computation complete.")

    # getting data ready for output.
    uk_passages = [uk_doc for uk_doc in list(top_matches.keys())]
    eu_passages = [eu_doc for eu_doc in list(top_matches.values())]
    similarity_scores = [score for score in list(sim_scores.values())]

    if invoke_llm_analysis:
        # get the model
        model_dict = get_model()
        model = model_dict['model']
        model_name = model_dict['name']

        """First LLM analysis."""
        # prepare the prompt template.
        prompt_file = file_handler.get_prompt_template(file_name='compare_text.txt')
        prompt_template = PromptTemplate.from_template(prompt_file)

        # instantiate model with prompt.
        llm_chain = LLMChain(prompt=prompt_template, llm=model)

        # run the llm on all passages.
        """THIS IS TEMPORARY CODE THAT NEEDS TO BE GENERALIZABLE FOR MORE THAN TWO PASSAGES"""
        # format data for output.
        llm_analyses = []

        print("Invoking LLM for Analysis")
        for i in range(0, len(uk_passages)):
            llm_analysis = llm_chain.invoke(
                prompt_inputs('passage1', uk_passages[i], 'passage2', eu_passages[i])
            )
            # As the llm_chain.invoke function returns a dict with the input variables and the text returned by llm,
            # we will only keep the text that is returned.
            llm_analyses.append(llm_analysis['text'])

        print("LLM Analysis complete")
        df = pd.DataFrame(
            {
                'eu_maternity_passage': eu_passages,
                'uk_maternity_passage': uk_passages,
                'similarity_score': similarity_scores,
                'llm_analysis': llm_analyses
            }
        )
        # only save if generation is not enabled to ensure two outputs are not produced.
        if not invoke_llm_generation:
            output_file_name = f'embeddings_comparator_{model_name}'
            file_handler.save_df_to_csv(df=df, file_name=output_file_name)

        if invoke_llm_generation:
            """Second LLM analysis."""
            # Get the prompt template for the second invocation of the llm.
            prompt_file2 = file_handler.get_prompt_template(file_name='generate_policy_guidance.txt')
            prompt_template2 = PromptTemplate.from_template(prompt_file2)

            # instantiate model with prompt.
            llm_chain = LLMChain(prompt=prompt_template2, llm=model)

            # run the llm for second layer of processing on all passages.
            llm_generation = []
            print("Invoking LLM for Generation")
            for i in range(0, len(uk_passages)):
                llm_analysis = llm_chain.invoke(
                    prompt_inputs('VERSION_1', uk_passages[i], 'VERSION_2', eu_passages[i])
                )
                # As the llm_chain.invoke function returns a dict with the input variables and the text returned by llm,
                # we will only keep the text that is returned.
                llm_generation.append(llm_analysis['text'])

            df = pd.DataFrame(
                {
                    'eu_maternity_passage': eu_passages,
                    'uk_maternity_passage': uk_passages,
                    'similarity_score': similarity_scores,
                    'llm_analysis': llm_analyses,
                    'generated_guidance': llm_generation
                }
            )

            output_file_name = f'embeddings_comparator_{model_name}'
            file_handler.save_df_to_csv(df=df, file_name=output_file_name)

    else:
        df = pd.DataFrame(
            {
                'eu_maternity_passage': eu_passages,
                'uk_maternity_passage': uk_passages,
                'similarity_score': similarity_scores
            }
        )

        output_file_name = f'embeddings_comparator'
        file_handler.save_df_to_csv(df=df, file_name=output_file_name)
