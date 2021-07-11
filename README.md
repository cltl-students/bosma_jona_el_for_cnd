This repository contains the code for the master thesis titled 'Entity Linking for Company Name Disambiguation', written by Jona B. Bosman, as part of the Linguistics master Text Mining.

Company names can be ambiguous: some companies have the same name and some companies go by several different names. The thesis proposes an Entity Linking system to disambiguate company names in Dutch news articles and link them to unique identifiers in a Knowledge Base.

The project was executed during an internship at [Brainial: Smart Tendering](https://brainial.com/). For additional information about the files described in this `README`, read the full thesis that included in the main directory of this repository: `entity_linking_for_company_name_disambiguation_2021.pdf`.

## Data
The data consists of Dutch news articles and a database of Dutch companies. All datafiles can be found in the data directory with a separate `README.md` that explains the structure of the directory. 

## How to run the project
To install all requirements to run the code for the project, run the following line in the terminal:\
`pip install -r requirements.txt`\
All code that was used in this project is included in the `src` directory.

### These scripts were run in this is the order to train the system and obtain the results reported in the thesis:

To execute all scripts in the correct order, execute `python main.py`.

`preprocessing.py` --> Preprocesses the datasets'.

`initial_kb.py` --> Creates an initial Knowledge Base with entity information and candidates for each company mention.

`annotation_preprocessing.py` --> Reforms the annotated data in the desired format and splits it into training, test and development data.

`iaa.py` --> Computes Inter-Annotator Agreement and the Cohen's Kappa on the data that was annotated by both annotators.

`probs_kb.py` --> Computes prior probabilities from the data and adds them to the Knowledge Base.

`training.py` --> Trains the system on the training data.

`evaluation.py` --> Evaluates the trained system and baselines on the test data.

`error_analysis.py` --> Performs an error_analysis on the system's output on the test data.

### This script was run to obtain the data from Brainial
`datascraper.py` --> Scrapes the news article and company data from Brainial using Elastic Search.

### These scripts were run in this order to prepare the data for annotation and run the Prodigy scripts to start annotating:
`data_preparation.py` --> Transforms the data in the right format to be annotated in Prodigy.

`iaa_annotations.py` --> Selects samples to be annotated by both annotators in order to compute Inter-Annotator Agreement and Cohen's Kappa, and transforms it in the right format to be annotated in Prodigy.

`prodigy_mult_cand.py` --> Starts the Prodigy environment for annotation for all data to be annotated.

`prodigy_iaa.py` --> Starts the Prodigy environment for annotation for the samples to be annotated by both annotators.

### This script was run to obtain statistics about the data:
`data_statistics.py` --> Computes and visualizes a number of statistics on the dataset.

## Resources
The `resources` directory contains the files that were, in addition to the data, needed to create and train the system. 

`kb_initial` --> The Knowledge Base containing all mentions in the data with their candidates, and all entities with their SBI-code descriptions.

`kb_probs` --> The updated Knowledge Base, now also containing the prior probabilities for each mention-entity pair that was included in the training data.

`nen_nlp` --> The NLP-object that is needed access the pipeline of spaCy, and trained to contain the Entity Linking system. This version of the NLP-object was custom trained by Brainial.

`nen_el_sentence` --> The NLP-object containing the Entity Linking system, trained on just the sentences of the mentions as context.

`nen_el_article` --> The NLP-object containing the Entity Linking system, trained on the full articles of the mentions as context.

## Results
The `results` directory holds two files that contain the results of the trained system on the evaluation set: `results.png` and `results.tsv`.

