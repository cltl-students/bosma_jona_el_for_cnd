import os
import csv
import spacy
import json
import random
import pickle
from pathlib import Path
from collections import Counter

from spacy.kb import KnowledgeBase
from spacy.util import minibatch, compounding
from sklearn.model_selection import train_test_split
from statistics import mean


def find_org_loc(doc, org):
    """Finds the location of a mention in a news article"""
    for ent in doc.ents:
        if ent.text == org:
            loc_begin = ent.start_char
            loc_end = ent.end_char
            return loc_begin, loc_end

    return None


def load_training_data(data_loc):
    """Loads and reformats the training data"""

    # Load nlp and prepare variables
    nlp = spacy.load('resources/nen_nlp')
    TRAIN_DOCS = []
    n_sents = []
    no_match = 0
    i = 0

    with open(data_loc, 'r', encoding='utf8') as infile:
        for line in infile:
            line = line.replace('\n', '').split('\t')

            # Skip the header line
            if line[0] != 'article':

                # Skip mentions that are labelled NIL
                if line[-1] != 'NIL':

                    # Print progress
                    if i % 100 == 0:
                        print(f"{i} samples preprocessed.")

                    # Extract values from data
                    text = line[0]
                    org = line[2]
                    kvk = line[-1]
                    offset = (line[3], line[4])

                    # Transform article into spaCy doc object
                    doc = nlp(text)
                    try:

                        # Save location of mention
                        loc_begin, loc_end = find_org_loc(doc, org)
                        offset = (loc_begin, loc_end)

                        # Save number of sentences per article
                        sent_len = len([sent for sent in doc.sents])
                        n_sents.append(sent_len)

                        # Create training instance
                        links_dict = {kvk: 1.0}
                        example = (doc, {"links": {offset: links_dict}})
                        TRAIN_DOCS.append(example)
                        i += 1
                    except:
                        no_match += 1
                        continue

    print(f"Number of samples skipped due to no match NER: {no_match}")
    print(f"Mean number of sentences per article: {mean(n_sents)}")
    print(f"Max number of sentences per article: {max(n_sents)}")
    print(f"Min number of sentences per article: {min(n_sents)}")

    return TRAIN_DOCS


def train_el():
    """Trains the Entity Linker on the training data"""

    # Load resources
    nlp = spacy.load('resources/nen_nlp')
    kb = KnowledgeBase(vocab=nlp.vocab, entity_vector_length=96)
    kb.load_bulk('resources/kb_probs')
    data_loc = "../data/model_data/train_data.tsv"

    # Format annotation results correctly
    TRAIN_DOCS = load_training_data(data_loc)
    print(len(TRAIN_DOCS), "training samples.")

    # Only train on ORG and NORP Named Entities.
    labels_discard = ['CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'MONEY', 'ORDINAL', 'PERCENT', 'PERSON', 'PRODUCT', 'QUANTITY', 'TIME', 'WORK_OF_ART']

    # Train Entity Linker pipeline
    entity_linker = nlp.create_pipe("entity_linker", config={"incl_prior": True,
                                                             'entity_vector_length': 96,
                                                             'n_sents': 0,
                                                             'labels_discard': labels_discard})
    entity_linker.set_kb(kb)
    nlp.add_pipe(entity_linker, last=True)

    print("Training the entity linker")

    # Train only the Entity Linker
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "entity_linker"]
    with nlp.disable_pipes(*other_pipes):  # train only the entity_linker
        optimizer = nlp.begin_training()

        # Iterate over training data
        for itn in range(100):
            random.shuffle(TRAIN_DOCS)
            batches = minibatch(TRAIN_DOCS, size=compounding(4.0, 32.0, 1.001))  # increasing batch size
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,
                    annotations,

                    # Prevent overfitting
                    drop=0.2,
                    losses=losses,
                    sgd=optimizer,
                )

            # Print progress and losses
            if itn % 1 == 0:
                print(itn, "Losses", losses)  # print the training loss
    print(itn, "Losses", losses)
    print()

    nlp.to_disk("resources/nen_nlp_el_sentence")


def main():
    train_el()


if __name__ == "__main__":
    main()