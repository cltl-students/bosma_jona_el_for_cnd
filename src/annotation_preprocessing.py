import pandas as pd
import json
import spacy
from sklearn.model_selection import train_test_split
import numpy as np


def find_org_loc(doc, org):
    """
    Function to get the start and end indices of the company mention in the news article

    :param doc:
    :param org:
    :return:
    """

    # Find all Named Entities in the article
    for sent in doc.sents:
        for ent in sent.ents:
            # Extract start and end indices of company mention
            if ent.text == org:
                loc_begin = ent.start_char
                loc_end = ent.end_char
                return loc_begin, loc_end, sent.text

    return None


def extract_annotations():
    """
    Prepares annotations from Prodigy to save them as a .tsv file

    :return: A dataframe with input data and labels
    :rtype: pandas.core.frame.DataFrame
    """

    # Prepare resources
    nlp = spacy.load("resources/nen_nlp")
    json_loc = "../data/prodigy_data/annotations+iaa_output.jsonl"

    # Prepare dict to store information in
    annotations = {'article':[], 'sent': [], 'small_context': [], 'org': [], 'loc_begin': [], 'loc_end': [], 'label': []}

    # Prepare variables
    i = 0
    mismatch = 0
    accepted = 0
    nil = 0
    o_articles = set()
    annotator_1 = 0
    used_annotator_1 = 0

    # Go through all annotations
    with open(json_loc, 'r', encoding='utf8') as jsonfile:
        for line in jsonfile:

            # Print progress
            if i%100 == 0:
                print(f"{i} samples preprocessed.")
            i += 1

            # Load annotation sample
            example = json.loads(line)

            # Count annotations per annotator
            if example['_session_id'] == "annotations3-Jona":
                annotator_1 += 1

            # Define context
            title = example['title']
            intro = example['intro']
            text_slice = example['slice']
            context = f"{title}. {intro}{text_slice}"

            # Extract full article for NER
            article = example['article']
            o_articles.add(article)
            article = article.replace('"', '')
            article = article.replace("'", "")

            org_text = example['org']

            # Find offset of organisation in context
            doc = nlp(article)
            if find_org_loc(doc, org_text):
                loc_begin, loc_end, sent = find_org_loc(doc, org_text)
            else:

                # Skip the sample if company mention is not recognised by spaCy's NER
                mismatch += 1
                print(f"{mismatch} NER mismatches.")
                continue

            # Save accepted answer
            if example['accept']:
                if example['accept'][0] not in ["NIL_notanorg", "NIL_otherentity", "NIL_ambiguous"]:
                    kvk = str(example['accept'][0])

                    # Samples were annotated with outdated KvK-numbers
                    if len(kvk) == 7:
                        kvk = '0'+kvk

                    accepted += 1

                    # Count number of samples one of the annotators did
                    if example['_session_id'] == "annotations3-Jona":
                        used_annotator_1 += 1

                    # Save info in a dict
                    annotations['article'].append(article)
                    annotations['sent'].append(sent)
                    annotations['small_context'].append(context)
                    annotations['org'].append(org_text)
                    annotations['loc_begin'].append(loc_begin)
                    annotations['loc_end'].append(loc_end)
                    annotations['label'].append(kvk)

                elif example['accept'][0] == 'NIL_otherentity':
                    nil += 1

    # Transform dict to pandas DataFrame
    annotations_df = pd.DataFrame.from_dict(annotations)
    a_articles = set(annotations_df['article'].unique())

    # Print statistics
    print()
    print(f"{accepted} mentions linked, {nil} NIL mentions.")
    print(f"Linked {annotations_df.shape[0]} of {i} mentions in {len(a_articles)} of {len(o_articles)} articles.")
    print()
    print(f"{annotator_1} samples annotated by annotator 1, {i-annotator_1} samples annotated by annotator 2.")
    print(f"{used_annotator_1} samples linked to entity by annotated by annotator 1, {accepted - used_annotator_1} samples linked to entity by annotator 2.")
    print()

    return annotations_df


def save_data(annotations_df):
    """
    Splits annotated data into training (60%), development (20%) and test (20%) data

    :param annotations_df: Dataframe with input samples and labels for training system
    :type annotations_df: pandas.core.frame.DataFrame
    """

    # Save all annotations in one file
    annotations_df.to_csv("../data/model_data/all_data.tsv", sep='\t', index=False)

    # Split annotations into training, dev and test data
    train_data, evalutation_data = train_test_split(annotations_df, test_size=0.40, random_state=1)
    test_data, dev_data = train_test_split(evalutation_data, test_size=0.5, random_state=1)

    # Get number of unique KvK-labels in each dataset
    train_unique_kvk = set(train_data['label'].unique())
    dev_unique_kvk = set(dev_data['label'].unique())
    test_unique_kvk = set(test_data['label'].unique())

    # Get number of unique company names each data
    train_unique_ment = set(train_data['org'].unique())
    dev_unique_ment = set(dev_data['org'].unique())
    test_unique_ment = set(test_data['org'].unique())

    # Get total number of articles in each dataset (articles can contain multiple company mentions)
    train_unique_art = set(train_data['article'].unique())
    dev_unique_art= set(dev_data['article'].unique())
    test_unique_art = set(test_data['article'].unique())

    # Save each dataset and print its statistics
    train_data.to_csv("../data/model_data/train_data.tsv", sep='\t', index=False)
    print(f"Training data with {len(train_unique_art)} articles and {train_data.shape[0]} samples. Saved in ../data/.")
    print(f"{len(train_unique_ment)} unique company mentions and {len(train_unique_kvk)} unique KvK-numbers.")
    print()

    dev_data.to_csv("../data/model_data/dev_data.tsv", sep='\t', index=False)
    print(f"Development data with {len(dev_unique_art)} articles and {dev_data.shape[0]} samples. Saved in ../data/.")
    print(f"{len(dev_unique_ment)} unique company mentions and {len(dev_unique_kvk)} unique KvK-numbers.")
    print()

    test_data.to_csv("../data/model_data/test_data.tsv", sep='\t', index=False)
    print(f"Test data with {len(test_unique_art)} articles and {test_data.shape[0]} samples. Saved in ../data/.")
    print(f"{len(test_unique_ment)} unique company mentions and {len(test_unique_kvk)} unique KvK-numbers.")
    print()


def main():
    annotations_df = extract_annotations()
    save_data(annotations_df)


if __name__ == '__main__':
    main()
