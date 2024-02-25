import yaml
from src.article_classifier import run_article_classifier
from src.embeddings_comparator import run_embeddings_comparison
from src.preprocess_pipeline import run_preprocess_pipeline
from src.text_comparator import run_text_comparator


def main():

    with open('configs/models_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    task = config['TASK'].lower()
    print("Running task: ", task)
    if task == 'article_classifier':
        run_article_classifier()
    elif task == 'embeddings_comparison':
        run_embeddings_comparison()
    elif task == 'preprocess_article':
        run_preprocess_pipeline(use_standard_cleaner=False, use_denoiser=True)
    elif task == 'text_comparator':
        run_text_comparator()

    print("Task Complete")


if __name__ == '__main__':
    main()
