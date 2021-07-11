import pickle
import spacy
from utils import *
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.metrics as sk
from spacy.kb import KnowledgeBase


def majority_baseline(test_data, kb):
    """Saves prediction from majority baseline on test data"""

    predictions = []
    for text, small_context, offset in test_data:

        # Extract company mentions and its candidates from the KB
        org = text[offset[0]:offset[1]]
        candidates = kb.get_candidates(org)

        # Prepare variables
        pred_prob = 0
        prediction = ""

        # Save the candidate with the highest probability
        for c in candidates:
            prob = c.prior_prob
            if prob > pred_prob:
                pred_prob = prob
                prediction = c.entity_

        predictions.append(prediction)

    return predictions


def system_predictions(test_data):
    """Get predictions on the test set from the trained model"""

    # Load resources and prepare variables
    nlp = spacy.load('resources/nen_nlp_el_sentence')
    predictions = []
    i = 0

    # Go through all test data samples
    for text, small_context, offset in test_data:
        i += 1

        # Make spaCy doc object from test data sample
        doc = nlp(text)

        # Extract org from text
        org = text[offset[0]:offset[1]]
        flag = False

        # Go through all Named Entities in the article
        for ent in doc.ents:

            # If entity
            if ent.text == org:
                predictions.append(ent.kb_id_)
                flag = True
                break

        # Mention is not recognised as Named Entity by spaC'y NER system
        if flag == False:
            predictions.append('NERror')

    return predictions


def get_prediction(org, vectorizer, clean_matrix, companies):
    """Find the KB entity with the highest name similarity to the company mention"""

    # Represent mention in the fitted TF-IDF n-gram vector
    dirty_matrix = vectorizer.transform([org])
    try:
        matches = awesome_cossim_top(dirty_matrix, clean_matrix.transpose(), 1, 0.74)
        non_zeros = matches.nonzero()
        sparsecols = non_zeros[1]

        # Return NIL if no matches can be found
        if len(sparsecols) < 1:
            return 'NIL'

        else:
            # Select the candidate with the highest similarity (= the first candidate)
            hit = companies.iloc[sparsecols[0], :]
            kvk = hit['kvk_number']

            return kvk

    except Exception as e:
        print(f"Failed to resolve org {org} with error: {e}")
        return None


def baseline_predictions(test_data):
    """Saves predictions of Brainial baseline"""

    # Loads data
    companies = pd.read_csv('../data/model_data/prepro_companies.tsv', sep='\t')
    companies['all_names'] = string_to_list(companies['all_names'])
    companies = companies.explode('all_names')

    # Prepare vectorizer
    vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams_chars, lowercase=False)
    clean_matrix = vectorizer.fit_transform(companies["all_names"])

    predictions = []

    # Save the KB entity with the highest name similarity to the company mention
    for text, small_context, offset in test_data:
        org = text[offset[0]:offset[1]]
        prediction = get_prediction(org, vectorizer, clean_matrix, companies)
        predictions.append(str(prediction))

    return predictions


def get_candidates(org, vectorizer, clean_matrix, companies):

    # Represent mention in the fitted TF-IDF n-gram vector
    dirty_matrix = vectorizer.transform([org])
    try:
        # Find the 5 KB entities that have the most similar TF-IDF vector, at least for 80%
        matches = awesome_cossim_top(dirty_matrix, clean_matrix.transpose(), 5, 0.8)
        non_zeros = matches.nonzero()
        sparsecols = non_zeros[1]

        if len(sparsecols) < 1:
            return None
        else:

            # Save info the selected candidates in a dictionary
            candidates = dict()
            for col, sim in zip(sparsecols, matches.data):
                hit = companies.iloc[col, :]
                kvk = hit['kvk_number']
                sbi = hit['sbi_code_description']
                city = hit['city']
                candidates[kvk] = {'sbi': f"{sbi} {city}", 'name_sim': sim}

            # Return the candidates
            return candidates

    except Exception as e:
        # print(f"Failed to resolve org {dirty_name} with error: {e}")
        return None


def context_prediction(candidates, text):
    """Selects candidate for a mention whose description fits the context best"""

    nlp = spacy.load('nl_core_news_lg')

    # Set best candidate to candidate with the highest fuzzy matching similarity
    #best_candidate = str(best_cand)
    #best_cand_doc = nlp(candidates[best_cand]['sbi'])

    # Retrieve doc-object of context
    context_doc = nlp(text)
    #highest_sim = best_cand_doc.similarity(context_doc)
    highest_sim = 0
    best_candidate = ""

    # Compare SBI-description doc to context doc for each candidate
    for candidate in candidates:

        # Get SBI description doc
        sbi_doc = nlp(candidates[candidate]['sbi'])

        # Compute Cosine similarity between vectors
        similarity = sbi_doc.similarity(context_doc)

        # Update best candidate/highest similarity variables
        if similarity > highest_sim:
            highest_sim = similarity
            best_candidate = candidate

    return best_candidate


def baseline_context_predictions(test_data):
    """Get predictions from Brainial baseline with context comparison"""

    # Load data
    companies = pd.read_csv('../data/model_data/prepro_companies.tsv', sep='\t')
    companies['all_names'] = string_to_list(companies['all_names'])
    companies = companies.explode('all_names')

    # Prepare vectorizer
    vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams_chars, lowercase=False)
    clean_matrix = vectorizer.fit_transform(companies["all_names"])

    predictions = []
    i = 0

    # Make prediction for each sample in test data
    for text, small_context, offset in test_data:

        # Print progress
        i += 1
        if i%50 == 0:
            print(f"{i} samples processed.")

        # Extract company mention
        org = text[offset[0]:offset[1]]

        # Get candidates for company mention
        candidates = get_candidates(org, vectorizer, clean_matrix, companies)

        # Select candidate whose SBI description matched the context best
        prediction = context_prediction(candidates, text)
        predictions.append(prediction)

    return predictions


def preprocess(test_loc):
    """Preprocesses test data"""

    # Prepare variables
    gold_labels = []
    test_data = []
    with open(test_loc, 'r', encoding='utf8') as infile:
        for line in infile:
            line = line.replace('\n', '').split('\t')

            # Skip header line and NIL labels
            if line[0] != 'article' and line[-1] != 'NIL':

                # Extract needed elements from test data
                text = line[0]
                small_context = line[1]
                loc_begin = int(line[3])
                loc_end = int(line[4])
                offset = (loc_begin, loc_end)
                gold_label = line[-1]

                # Append elements to the two lists
                gold_labels.append(gold_label)
                test_data.append((text, small_context, offset))

    return gold_labels, test_data


def evaluate():
    """Evaluates the trained model and three baseline systems on the test set"""

    # Load data and resources
    test_loc = "../data/model_data/test_data.tsv"
    nlp = spacy.load('resources/nen_nlp_el_sentence')
    kb = KnowledgeBase(vocab=nlp.vocab, entity_vector_length=96)
    kb.load_bulk('resources/kb_probs')

    # Preprocess test data in lists of samples and gold_labels
    gold_labels, test_data = preprocess(test_loc)

    # Transforms samples and gold_labels in dictionary
    predictions = dict()
    predictions['context'] = [context[0] for context in test_data]
    predictions['org'] = [context[0][context[2][0]:context[2][1]] for context in test_data]
    predictions['label'] = gold_labels

    # Print size of test data set
    print(f"Evaluated on {len(gold_labels)} company mentions.")
    print(f"{len(set(gold_labels))} unique KvK-numbers.")

    # Retrieve predictions on test set from all systems
    print("Getting system predictions...")
    system_preds = system_predictions(test_data)
    print("Getting baseline predictions...")
    baseline_preds = baseline_predictions(test_data)
    print("Getting baseline with context predictions...")
    base_context_preds = baseline_context_predictions(test_data)
    print("Getting majority baseline predictions...")
    majority_preds = majority_baseline(test_data, kb)


    # Print results for each system
    print()
    cr_system = sk.classification_report(gold_labels, system_preds, digits=3, output_dict=True, zero_division=False)
    print("System predictions:")
    print(cr_system['weighted avg'])

    print()
    cr_baseline = sk.classification_report(gold_labels, baseline_preds, digits=3, output_dict=True, zero_division=False)
    print("Baseline predictions")
    print(cr_baseline['weighted avg'])


    print()
    cr_base_context = sk.classification_report(gold_labels, base_context_preds, digits=3, output_dict=True, zero_division=False)
    print("Baseline+context predictions")
    print(cr_base_context['weighted avg'])


    print()
    cr_majority = sk.classification_report(gold_labels, majority_preds, digits=3, output_dict=True, zero_division=False)
    print("Majority baseline predictions")
    print(cr_majority['weighted avg'])

    # Save predictions in a .tsv file to be used in error analysis
    predictions['el_system'] = system_preds
    predictions_df = pd.DataFrame.from_dict(predictions)
    predictions_df.to_csv("../data/model_data/predictions.tsv", index=False, sep='\t')
    print("Saved predictions.tsv in ../data/model_data")

def main():
    evaluate()


if __name__ == "__main__":
    main()
