This directory contains the data that was created and used in this project. The data is structured into two directories: `project_data` and `prodigy_data`. Two files were compressed before they were added to this repository because their size was too big. These files are `nen_news.tsv.zip` and `annotations_input.jsonl.zip` and should be decompressed before use.

## Project data
`project_data` contains the following data files, needed to train and evaluate the system developed in this project.

`nen_news.tsv.zip` --> The raw news data, scraped from Brainial with Elastic Search. This file is compressed due to its big size and should be decompressed before use.

`nen_companies.tsv` --> The raw company data, scraped from Brainial with Elastic Search.

`prepro_news.tsv` --> The preprocessed news dataset.

`prepro_companies.tsv` --> The preprocessed company database.

`entities.tsv` --> File containing company name, unique identifier, SBI-code and SBI-code description for easy access.

`train_data.tsv` --> Part of the annotated data meant for training the system.

`dev_data.tsv` --> Part of the annotated data meant for developing the system.

`test_data.tsv` --> Part of the annotated data meant for evaluating the system.

`all_data.tsv` --> All annotated data.

`predictions.tsv` --> Predictions of the system on the test data to perform an error-analysis on.


## Prodigy Data
`prodigy_data` contains the following data files, needed as input for and retrieved as output from the Prodigy annotation environment.

`annotations_input.jsonl.zip` --> The news data in the right format to be annotated with Prodigy. This file is compressed due to its big size and should be decompressed before use. 

`annotations_output.jsonl` --> The annotated data.

`iaa_input.jsonl` --> The subset of the data to be annotated by two annotators in the right format for Prodigy.

`iaa_output.jsonl` --> The subset annotated by both annotators.

`annotations+iaa_output.jsonl` --> The concatenation of the regular annotated data and the samples from `iaa_output.jsonl` that were annotated the same by both annotators.
