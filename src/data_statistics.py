from utils import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import csv
import spacy
from spacy.kb import KnowledgeBase
import seaborn as sns


def count_mentions(articles, nlp):
    """Function to count the number of organization mentions in the news articles"""

    # Prepare variables
    n_articles = 0
    n_mentions = 0
    n_org_articles = 0
    n_org_mentions = 0

    i = 0

    # Go through all news articles
    for article in articles:
        i += 1

        # Pring progress
        if i%100 == 0:
            print(f"{i} articles counted.")

        # Transform article to spaCy doc
        doc = nlp(article)

        # Count all unique named entities
        entities = set([ent.text for ent in doc.ents])
        if entities:
            n_mentions += len(entities)
            n_articles += 1

        # Count all unique organizations
        org_entities = set([ent.text for ent in doc.ents if ent.label_ in ['ORG', 'NORP']])
        if org_entities:
            n_org_mentions += len(org_entities)
            n_org_articles += 1

    # Print statistics
    print(f"{i} articles in total.")
    print(f"{n_mentions} named entities in {n_articles} articles.")
    print(f"{n_org_mentions} campany mentions in {n_org_articles} articles.")


def get_statistics():
    """Function to load data and execute the count mentions function"""
    nlp = spacy.load('../resources/nen_nlp')
    news = pd.read_csv('../data/model_data/prepro_news.tsv', sep='\t')
    count_mentions(news['full_text'], nlp)


def get_distribution():
    """Function to plot the number times a KvK-number occurs in the data"""

    # Load data
    training_data = "../data/model_data/all_data.tsv"
    training_df = pd.read_csv(training_data, sep='\t')

    # Extract all KvK-numbers
    orgs = [org for org in training_df['label'] if org != 'NIL']
    #orgs = ['0'+org for org in orgs if len(org) == 7]

    # Count their occurrences and extract the 20 most common
    org_count = Counter(orgs)
    most_orgs = org_count.most_common(20)
    orgis = []
    percs = []

    # Save the first name of the companies that belong to the KvK-numbers
    entity_loc = "../data/model_data/entities.tsv"
    id_dict = dict()
    with open(entity_loc, "r", encoding="utf8") as csvfile:
        csvreader = csv.reader(csvfile, delimiter="\t")
        for row in csvreader:
            id_dict[row[0]] = (row[1], row[2], row[3])

    # Transform numbers into percentages
    for org, count in most_orgs:
        orgis.append(f"{id_dict[str(org)][0]} ({org})")
        perc = count/len(orgs)*100
        percs.append(perc)

    # Create barplot
    df = pd.DataFrame(list(zip(orgis, percs)),
                      columns=['Company', 'Percentage of annotations'])

    sns.barplot(y='Company', x='Percentage of annotations', data=df)
    plt.show()

    """
    sns.set(color_codes=True)
    x = 'org'

    (training_df
     .groupby(x)[y]
     .value_counts(normalize=True)
     .mul(100)
     .rename('percent')
     .reset_index()
     .pipe((sns.catplot, 'data'), x=x, y='percent', hue=y, kind='bar'))

    #plt.show() 
    """


def evaluation_graph():
    """Creates a plot showing the scores on the test set"""
    system = ["Entity Linker", "baseline", "baseline+context"]
    accuracy = [0.800, 0.599, 0.571]
    f_score = [0.379, 0.300, 0.289]

    x = np.arange(len(system))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, accuracy, width, label='micro F-score', color='#2e79ff')
    rects2 = ax.bar(x + width / 2, f_score, width, label='macro F-score', color='#abc9ff')

    hfont = {'fontname': 'Helvetica'}

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores', **hfont)
    ax.set_title('Results on development set', **hfont)
    ax.set_xticks(x)
    ax.set_ylim([0, 1])
    ax.set_xticklabels(system, **hfont)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    plt.show()


def n_candidates():
    """Creates a plot showing the number of samples and the number of candidates"""

    # Load data and resources
    data = "../data/model_data/all_data.tsv"
    data = pd.read_csv(data, sep='\t')
    orgs = data['org']
    nlp = spacy.load('resources/nen_nlp')
    new_kb = KnowledgeBase(vocab=nlp.vocab, entity_vector_length=96)
    new_kb.load_bulk('resources/kb_probs')

    # Count the number of candidates per alias
    n_cands = []
    for org in orgs:
        candids = new_kb.get_candidates(org)
        n_cands.append(len(candids))

    # Plot the number of mentions and the number of candidates
    data_candids = pd.DataFrame(n_cands, columns=['n_of_candidates'])

    sns.countplot(x='n_of_candidates', data=data_candids)
    plt.show()


def main():
    get_statistics()
    get_distribution()
    evaluation_graph()
    #n_candidates()


if __name__ == '__main__':
    main()
