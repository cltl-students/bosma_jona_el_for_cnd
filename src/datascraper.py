from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
import pandas as pd


def scrape_companies(es):
    """Function to scrape KvK numbers and additional information from companies"""

    # Prepare variables
    es_index = "nen-pilot-companies"
    nen_companies = []

    # Do not exlude any queries
    es_query = {"query": {"match_all": {}}}

    # Save every company entry from the scrape in a list
    for hit in scan(es, index=es_index, query=es_query):
        nen_companies.append(hit['_source'])

    # Transform result in pandas DataFrame
    nen_companies = pd.DataFrame(nen_companies)
    print(f"Scraped {len(nen_companies)} companies")

    return nen_companies


def scrape_news(es):
    """Function to scrape news articles and metadata"""

    # Prepare variables
    es_index = "nen-pilot-news"
    nen_news = []

    # Specify query for scraping the news articles
    es_query = {"query": {
        "bool": {
            "must": [],
            "filter": [
                {
                    "match_all": {}
                },
                {
                    "exists": {
                        # Take news articles that include organizations
                        "field": "resolved_orgs.keyword"
                    }
                },
                {
                    "exists": {
                        # Make sure the full text of the article is available
                        "field": "full_text",
                    }
                },
                {   # Make sure the title of the article is available
                    "exists": {
                        "field": "title",
                    }
                },
                {
                    "match_phrase": {
                        "language.keyword": {

                            # Take only Dutch articles
                            "query": "nl"
                        }
                    }
                },
                {
                    "range": {
                        "publish_date": {

                            # Take only recent articles
                            "format": "strict_date_optional_time",
                            "gte": "2021-01-06T16:16:38.151Z",
                            "lte": "2021-04-06T15:16:38.151Z"
                        }
                    }
                }
            ],
            "should": [],
            "must_not": []
        }}}

    # Add all relevant news articles to list
    for hit in scan(es, index=es_index, query=es_query):
        nen_news.append(hit['_source'])

    # Transform list into pandas DataFrame
    nen_news = pd.DataFrame(nen_news)
    print(f"Scraped {nen_news.shape[0]} news articles")

    return nen_news


def main():
    # Set up Elastic Search
    es = Elasticsearch(
        ["https://search.brainial.com/"],
        http_auth=("esuser", "ww_2020@12"),
        scheme="https",
        port=443,
    )

    # Scrape company information and save it as a tsv file
    nen_companies = scrape_companies(es)
    nen_companies.to_csv('../../data/model_data/nen_companies.tsv', sep='\t')

    # Scrape news articles and save them as a tsv file
    nen_news = scrape_news(es)
    nen_news.to_csv('../../data/model_data/nen_news.tsv', sep='\t')


if __name__ == '__main__':
    main()
