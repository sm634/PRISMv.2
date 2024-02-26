"""
Docstring
---------
Currently doing a single job for embedding red flag articles for human trafficing, so not very generalizable. Needs to
reuse components form embedding_funcs and models (for embeddings).
"""

import pandas as pd
from sentence_transformers import SentenceTransformer, util
from utils.preprocess_text import StandardTextCleaner
from collections import OrderedDict


def run_embeddings_comparison():
    indicators = {
        "redflag1": """A third party speaks on behalf of the customer (a third party may insist on being 
        present and/or translating).""",
        "redflag2": """A third party insists on being present for every aspect of the transaction.""",
        "redflag3": "A third party attempts to fill out paperwork without consulting the customer.",
        "redflag4": "A third party maintains possession and/or control of all documents or money.",
        "redflag5": "A third party claims to be related to the customer, but does not know critical details.",
        "redflag6": "A prospective customer uses, or attempts to use, third-party identification (of someone who is not"
                    "present) to open an account.",
        "redflag7": "A third party attempts to open an account for an unqualified minor.",
        "redflag8": "A third party commits acts of physical aggression or intimidation toward the customer.",
        "redflag9": """A customer shows signs of poor hygiene, malnourishment, fatigue, signs of physical and/or "
                    "sexual abuse, physical restraint, confinement, or torture.""",
        "redflag10": """A customer shows lack of knowledge of their whereabouts, cannot clarify where they live or
                              where they are staying, or provides scripted, confusing, or inconsistent stories in response
                              to inquiry.""",
        "redflag11": """Customers frequently appear to move through, and transact from, different geographic 
                  locations in the United States. These transactions can be combined with travel and 
                  transactions in and to foreign countries that are significant conduits for human trafficking.""",
        "redflag12": """Transactions are inconsistent with a customer’s expected activity and/or line of business 
                  in an apparent effort to cover trafficking victims’ living costs, including housing (e.g., hotel, motel, 
                  short-term rentals, or residential accommodations), transportation (e.g., airplane, taxi, limousine, or 
                  rideshare services), medical expenses, pharmacies, clothing, grocery stores, and restaurants, to include 
                  fast food eateries.""",
        "redflag13": """Transactional activity largely occurs outside of normal business operating hours 
                  (e.g., an establishment that operates during the day has a large number of transactions at night), 
                  is almost always made in cash, and deposits are larger than what is expected for the business and the 
                  size of its operations.""",
        "redflag14": """A customer frequently makes cash deposits with no Automated 
                  Clearing House (ACH) payments.""",
        "redflag15": """An individual frequently purchases and uses prepaid access cards.""",
        "redflag16": """A customer’s account shares common identifiers, such as a telephone number, email, and 
                  social media handle, or address, associated with escort agency websites and commercial sex advertisements.""",
        "redflag17": """Frequent transactions with online classified sites that are based in foreign jurisdictions.""",
        "redflag18": """A customer frequently sends or receives funds via cryptocurrency to or from darknet 
                  markets or services known to be associated with illicit activity. This may include services that host 
                  advertising content for illicit services, sell illicit content, or financial institutions that allow 
                  prepaid cards to pay for cryptocurrencies without appropriate risk mitigation controls.""",
        "redflag19": """Frequent transactions using third-party payment processors that conceal 
                  the originators and/or beneficiaries of the transactions.""",
        "redflag20": """A customer avoids transactions that require identification documents or that trigger 
                  reporting requirements."""}

    data = pd.read_csv('data/input/First200_ic.csv', encoding='latin-1')
    # instantiate text cleaner
    standard_cleaner = StandardTextCleaner()
    data['article'] = data['article'].apply(lambda x: standard_cleaner.remove_special_characters(x))
    data['id'] = data['ï»¿_id'].copy(deep=True)
    data = data.drop(labels=['ï»¿_id'], axis=1)

    articles = data.article.to_list()
    ids = data['id'].to_list()

    # instantiate the sentence transformer embeddings model.
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print("Instantiated Sentence Transformer Model")

    print("Creating embeddings for Articles.")
    # create embeddings for each article.
    articles_embeddings = {}
    for i in range(0, len(articles)):
        article_vector = model.encode(articles[i], convert_to_tensor=True)
        idx = ids[i]
        articles_embeddings[idx] = article_vector
    print("Articles Embeddings created.")

    print("Creating embeddings for Indicators.")
    # create embeddings for the indicators.
    indicators_embeddings = []
    # compute embedding for indicators
    for indicator in indicators:
        indicator_vector = model.encode(indicator, convert_to_tensor=True)
        indicators_embeddings.append(indicator_vector)
    print("Indicators Embeddings created.")

    print("Computing Cosine Similarity between articles and indicators")
    # calculate the similarity scores.
    top_indicators_articles = {}
    # calculate the similarity scores for each article and indicator and store the top_3.
    for idx in ids:
        article_vector = articles_embeddings[idx]
        # temp list
        top_three = []
        sim_scores = {}
        for i, indicator in enumerate(indicators):
            sim_scores[indicator] = util.pytorch_cos_sim(a=article_vector, b=indicators_embeddings[i])
            ordered_sim_scores = OrderedDict(sorted(sim_scores.items(),
                                                    key=lambda item: item[1],
                                                    reverse=True))
            top_three = list(ordered_sim_scores.items())[:3]

        top_indicators_articles[idx] = top_three
    print("Sematic Similarity computation complete.")

    article_indicators_df = pd.DataFrame(top_indicators_articles)
    article_indicators_df_transposed = article_indicators_df.transpose()
    article_indicators_df_transposed = article_indicators_df_transposed.reset_index()

    article_indicators_df_transposed.columns = ['article_id', 'indicator1', 'indicator2', 'indicator3']
    article_indicators_df_transposed = article_indicators_df_transposed.set_index('article_id')

    article_indicators_df_transposed.to_csv('data/output/articles_top_indicators.csv')
    print("Output Saved")
