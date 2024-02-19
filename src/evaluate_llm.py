"""
Doc String: Overview of metrics
-------------------------------
The documentation comes from https://docs.ragas.io/en/stable/getstarted/evaluation.html#get-started-evaluation

1. faithfulness - the factual consistency of the answer to the context base on the question.

2. context_precision - a measure of how relevant the retrieved context is to the question.
Conveys quality of the retrieval pipeline.

3. answer_relevancy - a measure of how relevant the answer is to the question

4. context_recall: measures the ability of the retriever to retrieve all the necessary
information needed to answer the question.
"""
from ragas.metrics import answer_relevancy, faithfulness
from ragas import evaluate
from datasets import Dataset
from langchain.embeddings import HuggingFaceEmbeddings
import pandas as pd

df = pd.read_csv('data/eval/template-generation-eval-dataset.csv')
dataset = Dataset().from_pandas(df)

result = evaluate(
    dataset=df,
    metrics=[
        answer_relevancy,
        faithfulness
    ]
)