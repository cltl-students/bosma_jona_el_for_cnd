import pandas as pd
import numpy as np
import re
from scipy.sparse import csr_matrix
import sparse_dot_topn.sparse_dot_topn as ct
from string import punctuation
import spacy
from collections import defaultdict

chars_to_remove = ['"', "'", "[", "]"]


def clean_element(element):
    """Removes unwanted characters from a text and preprocesses it."""

    # Remove characters
    for char in chars_to_remove:
        element = element.replace(char, '')

    # Further preprocessing
    element = element.strip()
    element = element.lower()

    return element


def string_to_list(series):
    series_list = []
    for elements in series:
        element_list = []
        for element in elements.split(','):
            cl_element = clean_element(element)
            element_list.append(cl_element)
        series_list.append(element_list)

    return series_list


def get_orgs(text, nlp):
    """

    :param text:
    :param nlp:
    :return: The entities that were extracted from the doc
    :rtype: list
    """
    orgs = []
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ['ORG', 'NORP']:
            if ent.text not in punctuation:
                org_texts = [ent.text for ent in orgs]
                if ent.text not in org_texts:
                    orgs.append(ent)

    return orgs


def get_orgs_sent(text, nlp):
    """

    :param text:
    :param nlp:
    :return: dictionary mapping recognised entity to the sentence it appears in
    :rtype: dict
    """
    doc = nlp(text)
    sents = [sent.text for sent in doc.sents]
    orgs_dict = defaultdict(list)
    for sent in sents:
        doc = nlp(sent)
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'NORP']:
                if ent.text not in punctuation:
                    orgs_dict[ent.text].append(sent)

    orgs_sents = dict()
    for org in orgs_dict:
        orgs_sents[org] = orgs_dict[org][0]

    return orgs_sents


def ngrams_chars(string, n=3):
    # string = fix_text(string)  # fix text encoding issues
    if pd.isna(string):
        string = ""

    string = string.encode("ascii", errors="ignore").decode()  # remove non ascii chars
    string = string.lower()  # make lower case
    chars_to_remove = [")", "(", ".", "|", "[", "]", "{", "}", "'"]
    rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
    string = re.sub(rx, '', string)  # remove the list of chars defined above
    string = string.replace('&', 'and')
    string = string.replace(',', ' ')
    string = string.replace('-', ' ')
    string = string.title()  # normalise case - capital at start of each word
    string = re.sub(' +', ' ', string).strip()  # get rid of multiple spaces and replace with a single space
    string = ' ' + string + ' '  # pad names for ngrams...
    string = re.sub(r'[,-./]|\sBD', r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    n_gramlist = [''.join(ngram) for ngram in ngrams]
    return n_gramlist


def awesome_cossim_top(A, B, ntop, lower_bound=0.0):
    # force A and B as a CSR matrix.
    # If they have already been CSR, there is no overhead
    A = A.tocsr()
    B = B.tocsr()
    M, _ = A.shape
    _, N = B.shape
    idx_dtype = np.int32
    nnz_max = M * ntop

    indptr = np.zeros(M + 1, dtype=idx_dtype)
    indices = np.zeros(nnz_max, dtype=idx_dtype)
    data = np.zeros(nnz_max, dtype=A.dtype)

    ct.sparse_dot_topn(
        M, N, np.asarray(A.indptr, dtype=idx_dtype),
        np.asarray(A.indices, dtype=idx_dtype),
        A.data,
        np.asarray(B.indptr, dtype=idx_dtype),
        np.asarray(B.indices, dtype=idx_dtype),
        B.data,
        ntop,
        lower_bound,
        indptr, indices, data)

    return csr_matrix((data, indices, indptr), shape=(M, N))


def resolve_org(dirty_name, vectorizer, clean_matrix, companies):
    dirty_matrix = vectorizer.transform([dirty_name])
    try:
        matches = awesome_cossim_top(dirty_matrix, clean_matrix.transpose(), 5, 0.8)
        non_zeros = matches.nonzero()
        sparsecols = non_zeros[1]

        if len(sparsecols) < 1:
            # print(f"Nothing detected for {dirty_name}")
            return None
        else:
            candidates = set()
            for col in sparsecols:
                hit = companies.iloc[col, :]
                kvk = hit['kvk_number']
                candidates.add(kvk)

            return candidates

    except Exception as e:
        # print(f"Failed to resolve org {dirty_name} with error: {e}")
        return None
