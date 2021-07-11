import pandas as pd
import spacy
from spacy.kb import KnowledgeBase


def entities_info(path):
    entity_info = dict()
    with open(path, 'r', encoding='utf8') as infile:
        for line in infile:
            row = line.split('\t')
            entity_info[row[0]] = dict()
            entity_info[row[0]]['name'] = row[1]
            entity_info[row[0]]['description'] = row[2]

    return entity_info


def error_analysis():
    nlp = spacy.load('../resources/nen_nlp')
    kb = KnowledgeBase(vocab=nlp.vocab, entity_vector_length=96)
    kb.load_bulk('../resources/kb_probs')

    predictions = pd.read_csv("../data/model_data/predictions.tsv", sep='\t')
    entity_info = entities_info("../data/model_data/entities.tsv")

    i = 0
    for prediction, label, org, sent in zip(predictions['el_system'], predictions['label'], predictions['org'], predictions['sentence']):
        label = str(label)
        if prediction != label and prediction != 'NIL':
            i += 1
            print()
            print(i, org)
            print([c.entity_ for c in kb.get_candidates(org)])
            print("Prediction:", entity_info[prediction]['name'], prediction)
            print(entity_info[prediction]['description'])
            print("Label:", entity_info[label]['name'], label)
            print(entity_info[label]['description'])
            print()
            print("Sentence: ", sent)
            print()

    print(i, "errors.")


def main():
    error_analysis()


if __name__ == "__main__":
    main()
