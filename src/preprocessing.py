import pandas as pd
from utils import string_to_list, clean_element


def get_intro(title, article):
    """
    Function to extract the first paragraph from a news article

    :param title: Title of the news article
    :param article: Full text of the article
    :return: the extracted first paragraph
    """

    # Split the article in paragraphs and remove empty paragraphs
    paraphs = article.split('\n\n')
    paraphs = [para for para in paraphs if para]

    # Extract the first paragraph, if it is not the same as the title
    for paraph in paraphs:
        if paraph.find(title) == -1:

            # Clean paragraph
            paraph = paraph.replace('\n\n', ' ')
            paraph = paraph.replace('\n', ' ')
            paraph = paraph.replace('  ', ' ')

            # Return paragraph
            return paraph

    # If article cannot be split in paragraphs, return empty string
    return ""


def merge_title_text(title, text):
    """Adds title of a text to the text itself."""

    # Clean article text
    text = text.replace('\n\n', ' ')
    text = text.replace('\n', ' ')
    text = text.replace('  ', ' ')

    # Try to locate title in the full text
    begin = text.find(title)

    # If the title is present, add a full stop to make it part of the full text
    if begin != -1:
        end = begin+len(title)
        new_text = f"{text[:end]}.{text[end:]}"

    # If the title is not present, add it to the text, with a full stop and space
    else:
        new_text = f"{title}. {text}"

    return new_text


def preprocess_orgs(orgs_series):
    """Reformats and cleans a series containing lists"""

    # Prepare container
    series = []
    for orgs in orgs_series:

        # Prepare list for each entry
        org_list = []
        for org in orgs:

            # Clean organization name and add it to org list
            clean_org = clean_element(org)
            org_list.append(clean_org)

        # Delete multiple occurrences of same org
        org_list = list(set(org_list))

        # Add all org lists together
        series.append(org_list)

    return series


def preprocess_news(news_path):
    """
    Function to preprocess the news database
    :param news_path: path to the raw news database
    :return: the preprocessed news database
    """

    # Read in news data and transform strings that are lists back to lists
    news = pd.read_csv(news_path, sep='\t')

    # Make sure every article has a title and the full text is accessible
    news.dropna(subset=['title', 'full_text', 'orgs'], inplace=True)
    news['orgs'] = string_to_list(news['orgs'])

    # Clean company names of unwanted characters
    news['orgs'] = preprocess_orgs(news['orgs'])

    # Find data span
    dates = news['extraction_timestamp']
    sorted_dates = dates.sort_values()
    print(f"Earliest date: {sorted_dates.iloc[0]}")
    print(f"Latest date: {sorted_dates.iloc[-1]}")

    # Merge title to the full article text
    news['intro'] = [get_intro(title, text) for title, text in zip(news['title'], news['full_text'])]
    news.dropna(subset=['intro'], inplace=True)
    news['full_text'] = [merge_title_text(title, text) for title, text in zip(news['title'], news['full_text'])]

    # Remove columns that are not used
    news.drop(news.columns.difference(['full_text', 'title', 'orgs', 'url', 'intro']), 1, inplace=True)

    return news


def merge_names(alt_names, first_names):
    """Adds first name of a company to a list of alternative names"""

    # Prepare container
    all_comp_names = []

    # Go through first name and alternative names of companies
    for alt_names, first_name in zip(alt_names, first_names):
        all_names = []

        # Clean first name and add it to all names list
        clean_first_name = clean_element(first_name)
        all_names.append(clean_first_name)

        # Clean each alternative name and add it to all names list
        for name in alt_names:
            clean_name = clean_element(name)
            all_names.append(clean_name)

        # Delete multiple occurrences of same names
        all_names = list(set(all_names))
        all_comp_names.append(all_names)

    return all_comp_names


def save_entities(companies):
    """
    Maps the names and SBI code descriptions to KvK-numbers
    of companies and saved them.
    """

    # Create dictionarie to map Kvk_numbers to company names and sbi code descriptions
    with open('../data/model_data/entities.tsv', "w", encoding="utf8") as csvfile:
        for kvk, name, desc, sbi in zip(companies.kvk_number, companies.name, companies.sbi_code_description, companies.sbi_code):
            row = f"{kvk}\t{name}\t{desc}\t{sbi}\n"
            csvfile.write(row)


def update_kvk(kvk):
    """
    Function to update outdated KvK-numbers
    :param kvk: the orginal KvK-number
    :return: the updated KvK-number, if it was updated
    """

    # Make sure KvK-number is a string
    kvk = str(kvk)

    # Add zero to beginning of outdated KvK-number and return it
    if len(kvk) == 7:
        new_kvk = '0'+kvk
        return new_kvk

    # If KvK-number is up to date, just return it
    else:
        return kvk


def preprocess_companies(companies_path):
    """
    Preprocessed companies database
    :param companies_path: Path to raw companies database
    :return: Preprocessed companies database
    """

    # Read in file and transform strings that are lists back to lists
    companies = pd.read_csv(companies_path, sep='\t')

    # Update outdated KvK-numbers
    companies['kvk_number'] = companies['kvk_number'].apply(update_kvk)

    companies['alternative_names'] = string_to_list(companies['alternative_names'])

    # Make sure all companies have a sbi_code_description
    companies.dropna(subset=['sbi_code_description'], inplace=True)

    # Add main company name to alternative names and save it as all_names
    companies['all_names'] = merge_names(companies['alternative_names'], companies['name'])

    # Remove columns that are not used
    companies.drop(companies.columns.difference(['kvk_number', 'name', 'all_names', 'sbi_code_description', 'sbi_code', 'city']),
                   1,
                   inplace=True)

    return companies


def main():
    # Preprocess news data and save it
    news_path = "../data/model_data/nen_news.tsv"
    prepro_news = preprocess_news(news_path)
    prepro_news.to_csv('../data/model_data/prepro_news.tsv', sep='\t')

    # Preprocess company data and save it
    companies_path = "../data/model_data/nen_companies.tsv"
    prepro_companies = preprocess_companies(companies_path)
    save_entities(prepro_companies)
    prepro_companies.to_csv('../data/model_data/prepro_companies.tsv', sep='\t', index=False)


if __name__ == '__main__':
    main()
