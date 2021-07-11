from spacy.kb import KnowledgeBase
from utils import *
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer


def add_entities(kb, desc_dict, nlp):
    """
    Adds company entities to KB

    :param kb: the empty Knowledge Base
    :param desc_dict: dict with KvK-numbers and SBI-code descriptions of companies
    :param nlp: the nlp object to retrieve vector of SBI-code description
    :return: the KB with companies entities added
    """

    # Add entities (set_entities)
    print("Adding entities to KB...")
    for n, (kvk, desc) in enumerate(desc_dict.items()):
        if n % 1000 == 0:
            print(f"{n}/{len(desc_dict)} entities added.")

        desc_doc = nlp(desc)
        desc_enc = desc_doc.vector
        kb.add_entity(entity=str(kvk), entity_vector=desc_enc, freq=1)

    print("Done adding entities!")

    return kb


def find_candidates(companies, news,  nlp):
    """
    Function to find candidates for each company mention in the news articles

    :param companies: the database with company entities
    :param news: the news database
    :param nlp: spaCy nlp object to perform NER
    :return: a dictionary with mentions as keys and their candidates as values
    """

    # Prepare variables for mention detection
    companies = companies.explode('all_names')
    n_articles = news['full_text'].shape[0]
    mention_cands = defaultdict(list)
    articles_with_mentions = 0
    n_mentions = 0

    # Find candidates for entity mentions
    print()
    print("Finding candidates for mentions...")
    for n, article in enumerate(news['full_text']):
        flag = False
        if n % 10 == 0:
            print(f"{n}/{n_articles} processed.")
            print(f"{len(mention_cands)} mentions with candidates")

        # Prepare vectorizer
        vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams_chars, lowercase=False)
        clean_matrix = vectorizer.fit_transform(companies["all_names"])

        # Extract Named Entities from the article
        mentions = [ent.text for ent in get_orgs(article, nlp)]
        n_mentions += len(mentions)

        # Find candidates for each entity mentions in the articles
        for company in mentions:

            # Only find candidates for mentions that don't already have candidates
            # Different articles can contain the same mentions
            if company not in mention_cands:

                # Extrac the candidates
                candidate_comps = resolve_org(company, vectorizer, clean_matrix, companies)

                # Add mentions with candidates to dictionary
                if candidate_comps:
                    flag = True
                    for kvk in candidate_comps:
                        mention_cands[company].append(str(kvk))

        # Count articles that contain mentions
        if flag:
            articles_with_mentions += 1

    print()
    print("Total number of org/norp mentions in articles:")
    print(f"{n_mentions} ORG/NORP mentions")
    print("Total number of articles with mentions with candidates:")
    print(f"{articles_with_mentions} articles.")
    print("Total number of unique mentions with candidates:")
    print(len(mention_cands))

    return mention_cands


def add_aliases(mention_cands, kb):
    """
    Add all mentions with candidates to the Knowledge Base as aliases

    :param mention_cands: the dictionary with mentions and their candidates
    :param kb: the Knowledge Base that contains the company entities
    :return: the Knowledge Base with the added aliases
    """

    print()
    print("Adding aliases to knowledge base...")

    # Go through all mentions in the mention/candidate dict
    for n, mention in enumerate(mention_cands):

        # Print progress
        if n % 10 == 0:
            print(f"{n}/{len(mention_cands)} added")

        # Find candidates and prior probabilities
        candidates = mention_cands[mention]
        prob = round(1 / len(candidates), 1)

        # Assume prior probability is the same for each candidate
        # (This will be updated after the annotation)
        probabilities = [prob] * len(candidates)

        # Add each alias to the Knowledge Base
        kb.add_alias(mention, candidates, probabilities)

    # Print size of KB
    n_aliases = kb.get_size_aliases()
    print(f"{n_aliases} aliases added.")

    return kb


def create_kb():

    # Load datasets
    companies = pd.read_csv('../data/model_data/prepro_companies.tsv', sep='\t')
    companies['all_names'] = string_to_list(companies['all_names'])
    news = pd.read_csv('../data/model_data/prepro_news.tsv', sep='\t')
    news['orgs'] = string_to_list(news['orgs'])
    nlp = spacy.load('../resources/nen_nlp')

    # Create dictionaries to map Kvk_numbers to company names and sbi code descriptions
    name_dict = dict(zip(companies.kvk_number, companies.name))
    desc_dict = dict(zip(companies.kvk_number, companies.sbi_code_description))

    # Load NLP pipeline and Knowledge Base
    kb = KnowledgeBase(vocab=nlp.vocab, entity_vector_length=96)

    # Add entities to Knowledge Base and save it
    kb = add_entities(kb, desc_dict, nlp)
    kb.dump("../resources/kb_entities")

    # Find candidates for each mention in the news data
    mention_cands = find_candidates(companies, news, name_dict)

    # Add aliases for all mentions with candidates to Knowledge Base
    kb = add_aliases(mention_cands, kb)

    # Save Knowledge Base
    kb.dump("../resources/kb_initial")

    # Save NLP (needed to store vectors of company mentions found in the news articles)
    nlp.to_disk("../resources/nen_nlp")


def main():
    create_kb()


if __name__ == "__main__":
    main()
