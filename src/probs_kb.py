import spacy
from spacy.kb import KnowledgeBase
from collections import defaultdict


def get_prior_probs(candidates, alias_dict):
    """
    Get prior probabilities of all candidates for a mention

    :param candidates: the list of candidates for a company mention
    :param alias_dict:
    :return:
    """

    prior_probs = []
    candids = []

    total = sum(alias_dict[cand] for cand in alias_dict)

    for cand in candidates:
        if cand in alias_dict:
            prob = alias_dict[cand] / total
        else:
            prob = 0

        candids.append(cand)
        prior_probs.append(prob)

    return candids, prior_probs


def add_aliases(cands_dict, old_kb, new_kb):
    for alias in old_kb.get_alias_strings():
        candids = old_kb.get_candidates(alias)
        candidates = [cand.entity_ for cand in candids]

        if alias in cands_dict:
            candidates, prior_probs = get_prior_probs(candidates, cands_dict[alias])
            print(prior_probs)

        #else:
            #prior_probs = [old_kb.get_prior_prob(cand.entity_, alias) for cand in candids]

            new_kb.add_alias(alias, candidates, prior_probs)

    return new_kb


def save_candidates(datapath):
    """


    :param datapath: path to all data
    :return:
    """
    cands_dict = dict()
    i = 0
    with open(datapath, 'r', encoding='utf8') as infile:
        for line in infile:
            line = line.replace('\n', '').split('\t')
            if line[0] != 'context' and line[-1] != 'NIL':
                if i % 100 == 0:
                    print(f"{i} samples preprocessed.")
                i += 1
                alias = line[2]
                entity = line[-1]

                if alias not in cands_dict:
                    cands_dict[alias] = defaultdict(int)

                cands_dict[alias][entity] += 1

    return cands_dict


def redefine_kb():
    """


    :return:
    """

    # Preprare resources
    nlp = spacy.load('resources/nen_nlp')
    old_kb = KnowledgeBase(vocab=nlp.vocab, entity_vector_length=96)
    old_kb.load_bulk('resources/kb_initial')

    # Create new Knowledge Base, with the entities from the comapany database
    new_kb = KnowledgeBase(vocab=nlp.vocab, entity_vector_length=96)
    new_kb.load_bulk('resources/kb_entities')

    # Load data
    datapath = "../data/model_data/all_data.tsv"

    # Find candidates and number of occurrences
    cands_dict = save_candidates(datapath)

    # Add aliases to KB
    new_kb = add_aliases(cands_dict, old_kb, new_kb)
    print(f"Added {new_kb.get_size_aliases()} aliases to KB and their prior probabilities.")

    # Save new KB
    new_kb.dump("../resources/kb_probs")


def main():
    redefine_kb()


if __name__ == "__main__":
    main()
