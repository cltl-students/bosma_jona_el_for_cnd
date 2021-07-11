import json
import jsonlines
import spacy
from spacy.kb import KnowledgeBase


def save_500():
    # Prepare datafiles
    json_loc = "../../data/prodigy_data/annotations_input.jsonl"
    new_loc = "../../data/prodigy_data/iaa_input.jsonl"

    # Prepare resources
    nlp = spacy.load('../resources/nen_nlp')
    kb = KnowledgeBase(vocab=nlp.vocab, entity_vector_length=96)
    kb.load_bulk('../resources/kb_initial')

    i = 0
    j = 0
    unique_orgs = []
    limit = 400

    # Open file to save IAA-annotations in
    outfile = jsonlines.open(new_loc, 'w')

    # Go through all annotations
    with open(json_loc, 'r', encoding='utf8') as jsonfile:
        for line in jsonfile:
            example = json.loads(line)
            org = example['org']
            if len(kb.get_candidates(org)) > 1:
                i += 1
                if i > 4070 and org not in unique_orgs and j < limit:
                    j += 1
                    outfile.write(example)
                    unique_orgs.append(org)
                    print(j, ", sample: ", i)

    outfile.close()
    print(f"{limit} IAA-annotations Prodigy input saved in ../prodigy/iaa_input.jsonl")


def main():
    save_500()


if __name__ == '__main__':
    main()
