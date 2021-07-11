import pandas as pd
import jsonlines
from utils import string_to_list
import spacy
from spacy.kb import KnowledgeBase
import re
from utils import get_orgs


def highlight(text, org):
    """
    Highlights the company mention in the text presented during annotation

    :param text: the text to be presented
    :param org: the mention to be highlighted
    :return: HTML object of text with highlighted company
    """

    # Find locations of the company
    occurrences = [m.start() for m in re.finditer(org, text)]

    # Add highlight to each occurrence of the company
    for n, begin in enumerate(occurrences):
        begin += 13*n
        end = begin+len(org)
        text = f"{text[:begin]}<mark>{text[begin:end]}</mark>{text[end:]}"

    return text


def org_mention(org, text, size=500):
    """
    Extracts the portion of the article in which the company mention occured first

    :param org: the company mention
    :param text: the full company text
    :param size: the desired character size of the portion
    :return: the text slice in which the company mention occurred first
    """

    # Number of characters to include before and after the company mention
    length = size//2

    # Find the index of the company mention in the news article
    begin = text.find(org)

    # Select the index of the begin and end of the text slice
    begin_slice = begin-length
    end_slice = begin+length

    # Adjust indices if they exceed beginning or end of article
    if end_slice > len(text):
        end_slice = len(text)
    if begin_slice < 0:
        begin_slice = 0

    # Take the slice
    text_slice = text[begin_slice:end_slice]

    return '...'+text_slice+'...'


def sbi_addition(sbi_code):
    """Creates link to SBI-code explanation."""
    base = "https://sbi.cbs.nl/cbs.typeermodule.typeerservicewebapi/content/angular/app/#/code?sbicode="
    link = base+str(sbi_code)

    return link


def prodigy_input():
    """Transforms data to be annotated in right format for the Prodigy annotation environment."""

    # Load news article data
    data_path = "../data/model_data/prepro_news.tsv"
    news = pd.read_csv(data_path, sep='\t')

    # Load resources
    nlp = spacy.load("resources/nen_nlp")
    kb = KnowledgeBase(vocab=nlp.vocab, entity_vector_length=300)
    kb.load_bulk("resources/kb_initial")

    # Extract all aliases from the Knowledge Base
    kb_aliases = kb.get_alias_strings()

    # Prepare variables
    counter = 0
    article_count = 0
    mult_cands = 0
    mult_cand_articles = 0

    # Save all annotation samples as lines in a JSON lines files
    with jsonlines.open('../data/prodigy_data/annotations_input.jsonl', "w") as writer:

        # Go through all news data
        for n, (title, article, url, intro) in enumerate(zip(news['title'],
                                              news['full_text'],
                                              news['url'],
                                              news['intro'])):

            # Extract company mentions
            orgs = get_orgs(article, nlp)

            # Set flags
            mult_cand_flag = False
            article_flag = False

            # Extract elements for each company mention in the news article
            for org in orgs:

                # Check if company name is in the Knowledge Base
                if org.text in kb_aliases:
                    article_flag = True

                    # Print progress
                    if counter % 10 == 0:
                        print(f"{counter} company mentions saved for annotation.")

                    # Select only company mentions with multiple candidates and count them for statistics
                    if len(kb.get_candidates(org.text)) > 1:
                        mult_cands += 1
                        mult_cand_flag = True

                    # Save full article if it is shorter than 1000 characters
                    if len(article) < 1000:

                        # Highlight the company mention
                        h_article = highlight(article, org.text)
                        displayed_text = h_article

                    # Extract relevant portions for articles longer than 1000 characters
                    else:
                        # Check if the first paragraph/introduction is available
                        if type(intro) != str:
                            continue

                        # Highlight company mentions in introduction if possible
                        try:
                            h_intro = highlight(intro, org.text)
                        except:

                            # If not possible, save the unhighlighted texts
                            print("Intro:", intro)
                            print("Org:", org.text)
                            continue

                        # Extract slice where company mention occurs and highlight it
                        slice_text = org_mention(org.text, article)
                        h_slice_text = highlight(slice_text, org.text)

                        # Combine all text elements into article text to be displayed during annotation
                        displayed_text = f"<b>Inleiding</b>\n{h_intro}\n\n{h_slice_text}"

                    # Save URL to read article
                    article_link = f"<a href='{url}' target='_blank'>Lees hier het hele artikel.</a>"

                    # Add URL to displayed article text
                    text = f"Bedrijf: <b>{org}</b>\n\n{title}\n\n{displayed_text}\n\n{article_link}"
                    context = f"{title}{displayed_text}"

                    # Extract location of the company mention
                    loc_begin = org.start_char
                    loc_end = org.end_char

                    # Write row with all elemtns into the JSON lines file
                    row = {'html': text, 'org': org.text, 'context': context, 'article': article, 'title': title, 'intro': intro, 'slice': slice_text, 'loc_begin': loc_begin, 'loc_end': loc_end}
                    writer.write(row)

                    # Count annotation samples
                    counter += 1

            if article_flag:
                article_count += 1
            if mult_cand_flag:
                mult_cand_articles += 1

    # Print statistics
    print()
    print(f"Saved {counter} companies from {article_count} articles for annotating.")
    print(f"Found {mult_cands} multi-candidate mentions in {mult_cand_articles} articles.")
    print(f"Found {counter-mult_cands} one-candidate mentions in {article_count-mult_cand_articles} articles.")


def main():
    prodigy_input()


if __name__ == "__main__":
    main()
