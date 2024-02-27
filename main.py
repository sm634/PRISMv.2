from src.article_classifier import run_article_classifier
from src.article_redflag_comparator import run_article_redflag_comparator
from src.preprocess_pipeline import run_preprocess_pipeline
from src.text_comparator import run_text_comparator
from src.embeddings_comparator import run_embeddings_comparator

from utils.files_handler import FileHandler

file_handler = FileHandler()


def main():
    # get models config
    file_handler.get_config()
    config = file_handler.config
    # get arguments config
    file_handler.get_config('arguments_passer.yaml')
    arguments_config = file_handler.config

    task = config['TASK'].lower()
    print("Running task: ", task)
    if task == 'article_classifier':
        run_article_classifier()
    elif task == 'redflag_article_comparator':
        run_article_redflag_comparator()
    elif task == 'preprocess_article':
        run_preprocess_pipeline(use_standard_cleaner=False, use_denoiser=True)
    elif task == 'text_comparator':
        run_text_comparator()
    elif task == 'embeddings_comparator':
        llm_analysis = arguments_config['INVOKE_LLM_ANALYSIS']
        llm_generation = arguments_config['INVOKE_LLM_GENERATION']
        run_embeddings_comparator(invoke_llm_analysis=llm_analysis,
                                  invoke_llm_generation=llm_generation)

    print("Task Complete")


if __name__ == '__main__':
    main()
