"""
Custom Prodigy recipe to perform manual annotation of entity links,
given an existing NER model and a knowledge base performing candidate generation.
"""

import spacy
from spacy.kb import KnowledgeBase

import prodigy
import random
from prodigy.models.ner import EntityRecognizer
from prodigy.components.loaders import JSONL
from prodigy.util import set_hashes
from prodigy.components.filters import filter_duplicates

import csv
from pathlib import Path

# Define Prodigy recipe
@prodigy.recipe(
    "multiple_candidates",
    dataset=("The dataset to use", "positional", None, str),
    source=("The source data as a .txt file", "positional", None, Path),
    nlp_dir=("Path to the NLP model", "positional", None, Path),
    kb_loc=("Path to the KB", "positional", None, Path),
    entity_loc=("Path to the file with additional information about the entities", "positional", None, Path)
)
def multiple_candidates(dataset, source, nlp_dir, kb_loc, entity_loc):

    # Initialize the Prodigy stream by running the NER model
    nlp = spacy.load(nlp_dir)
    kb = KnowledgeBase(vocab=nlp.vocab, entity_vector_length=1)
    kb.load_bulk(kb_loc)

    # Read the pre-defined CSV file into dictionaries mapping QIDs to the full names and descriptions
    id_dict = dict()
    with entity_loc.open("r", encoding="utf8") as csvfile:
        csvreader = csv.reader(csvfile, delimiter="\t")
        for row in csvreader:
            id_dict[row[0]] = (row[1], row[2], row[3])

    # Define stream of examples
    stream = JSONL(source)
    stream = _add_options(stream, kb, id_dict)

    return {
        "dataset": dataset,
        "stream": stream,
        "view_id": "choice",
        'config': {'feed_overlap': False}
    }


def _add_options(stream, kb, id_dict):
    """Create options for annotation"""

    # Add options to choose from for annotators
    for task in stream:

        # Extract candidates for each company mention
        mention = task['org']
        candidates = kb.get_candidates(mention)

        # Take only company mentions with multiple candidates
        if len(candidates) > 1:
            options = [{"id": c.entity_, "html": _print_info(c.entity_, id_dict)} for c in candidates]
            options.append({"id": "NIL_notanorg", "text": "Herkende 'bedrijf' is geen bedrijf."})
            options.append({"id": "NIL_otherentity", "text": "Bedrijf staat niet in de opties."})
            options.append({"id": "NIL_ambiguous", "text": "Niet genoeg context."})

            task["options"] = options

            yield task


def _print_info(kvk, id_dict):
    """ For each candidate company entity, create a links to websites with extra information them """

    # Get name, SBI-code and SBI-code description from the id_dict for the KvK-number of each candidate
    name = id_dict[kvk][0]
    desc = id_dict[kvk][1]
    sbi_code = id_dict[kvk][2]
    sbi = str(int(float(sbi_code)))

    # Create URL to CBS page with explanation of SBI-code
    base_sbi_url = "https://sbi.cbs.nl/cbs.typeermodule.typeerservicewebapi/content/angular/app/#/code?sbicode="
    sbi_url = base_sbi_url+sbi
    sbi_url = f"<a href='{sbi_url}' target='_blank'>{desc}</a>"

    # Create URL to query the Kamer van Koophandel Handelsregister with the KvK-number of the candidate
    kvk_link = "https://www.kvk.nl/zoeken/handelsregister/?kvknummer=" + kvk
    kvk_url = f"<a href='{kvk_link}' target='_blank'>{kvk}</a>"

    option = f"{name}: {sbi_url} (KvK: {kvk_url})"
    return option


"""
Copy this command in the terminal to start Prodigy's annotation environment
prodigy multiple_candidates annotation_results ../../data/prodigy_data/annotations_input.jsonl resources/nen_nlp resources/kb_initial ../../data/model_data/entities.tsv -F prodigy_annotation.py

Copy this command in the terminal to save the results of the annotation to the local machine
prodigy db-out annotation_results >> ../../data/prodigy_data/test_results2.jsonl
"""