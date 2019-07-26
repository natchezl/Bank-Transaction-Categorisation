import pandas as pd
import spacy

from utils import misc_utils


def preprocess_description(df, remove_var=False):
    df['desc'] = df['Description']
    df['Description'] = mask_descriptions(df['Description'])
    df = extract_description_features(df, 'Description', remove_var=remove_var)
    df['Description'] = remove_descriptions(df['Description'])
    df['Description'] = tokenize_lemmatize(df['Description'])
    return df


def mask_descriptions(column):
    column = mask_acct_no(column)
    column = mask_foreign_currency(column)
    # column = mask_foreign_country(column)
    return column


def extract_description_features(df, desc_col_name, remove_var=False):
    """
    Extract $ACCT_NO, $CURRENCY and $FOREIGN_COUNTRY from description to a new column
    :param df: pd.DataFrame contains data
    :param desc_col_name: str, description column name
    :param remove_var: boolean, if True, $ACCT_NO, $CURRENCY and $FOREIGN_COUNTRY will be removed from the description column
    :return: df with extended column ['isAcctNo', 'isForeignCurr', 'isForeignCountry']
    """
    df['isAcctNo'] = df[desc_col_name].str.contains('$ACCT_NO', regex=False).astype('int')
    df['isForeignCurr'] = df[desc_col_name].str.contains('$CURRENCY', regex=False).astype('int')
    # df['isForeignCountry'] = df[desc_col_name].str.contains('$FOREIGN_COUNTRY', regex=False).astype('int')

    if remove_var:
        regex = '\$ACCT_NO|\$CURRENCY|\$FOREIGN_COUNTRY'
        df[desc_col_name] = df[desc_col_name].str.replace(regex, '', regex=True)
    return df


def remove_descriptions(column):
    column = remove_suburb_names(column)
    column = remove_stopwords(column)
    column = remove_numbers(column)
    column = remove_punctuation(column)
    column = column.str.strip()
    return column


def tokenize_lemmatize(column, min_word_len=2):
    """
    Tokenize and Lemmatize text
    :param column:
    :param min_word_len: minimum characters in each word
    :return: column contains list of tokenized words
    """
    nlp = spacy.load('en', disable=['tagger', 'parser', 'ner'])
    docs = column.tolist()

    def token_filter(token):
        return not (token.is_punct | token.is_space) and (len(token.text) >= min_word_len)

    filtered_tokens = []
    for doc in nlp.pipe(docs):
        tokens = [token.lemma_ for token in doc if token_filter(token)]
        filtered_tokens.append(tokens)

    return filtered_tokens


def remove_suburb_names(column):
    # remove suburb_names
    suburbs = misc_utils.load_suburb_names()
    suburbs = '|'.join(suburbs[0])
    return column.str.replace(suburbs, ' ', regex=True)


def remove_stopwords(column):
    stopwords = misc_utils.load_stopwords()
    # replace only words with white spaces
    stopwords[0] = r'\b' + stopwords[0] + r'\b'
    stopwords = '|'.join(stopwords[0])
    return column.str.replace(stopwords, ' ', regex=True)


def remove_numbers(column):
    return column.str.replace(r'\d+(.\d+)?', ' ')


def remove_punctuation(column):
    return column.str.replace(r'[\-*:()@<>#]', ' ')


def mask_acct_no(column):
    """
    Mask account number with $ACCT_NO
    :param column:
    :return:
    """
    return column.str.replace(r'\d*\*{3,}\d*|\d+(\-\d+){2,}', ' $ACCT_NO ')


def mask_foreign_currency(column):
    """
    Mask foreign currency text with $CURRENCY
    ** Must do before removing numbers from the text
    :param column:
    :return:
    """
    curr = misc_utils.load_currencies()
    # Drop New Zealand currencies
    curr = curr.drop(curr[curr['id'] == 'nzd'].index)
    # Drop ancient currencies
    curr = curr.drop(curr[curr.value.str.contains(r'\(\d', regex=True)].index)
    # Remove texts in brackets: belgian franc (convertible) -> belgian franc
    curr['value'] = curr['value'].str.split('\(').str[0].str.strip()
    # Build regex: british pound -> (british)? pound
    curr['value'] = curr['value'].replace({r'(^.+)+( \w*$)': r'\d+(\.\d*)? ?(\1)? ?\2(s)?\\b'}, regex=True)
    # Match ex. 12usd | 12 usd | <\b>usd<\b>
    curr['id'] = r'(\d+(\.\d*)? ?| )' + curr['id'] + r'\b'
    # replace both currency code & currency name
    curr_regex = '|'.join(curr['value'])
    curr_regex = '|'.join(curr['id']) + curr_regex
    column = column.replace(curr_regex, ' $CURRENCY ', regex=True)
    return column


def mask_foreign_country(column):
    """
    Mask foreign country name with $FOREIGN_COUNTRY
    :param column:
    :return:
    """
    codes = misc_utils.load_country_code()
    # Remove New Zealand from foreign country list
    codes = codes.drop(codes[codes['Alpha-2'] == 'nz'].index)
    # Remove texts in brackets: belgian franc (convertible) -> belgian franc
    codes['Country'] = codes['Country'].replace({r'\(.*\)': ''}, regex=True).str.strip()
    regex = list()
    regex.append('|'.join(r'\s' + codes['Country'] + r'\b'))
    # Don't use Alpha-2 and Alpha-3 since there are lots of misreplacement
    # regex.append('|'.join(r'\s' + codes['Alpha-2'] + r'\b'))
    # regex.append('|'.join(r'\s' + codes['Alpha-3'] + r'\b'))
    regex_str = '|'.join(regex)
    column = column.replace(regex_str, ' $FOREIGN_COUNTRY ', regex=True)
    return column


def get_most_freq_words(countvec_transformed, countvec):
    sum_words = countvec_transformed.sum(axis=0)
    words_freq = dict((word, sum_words[0, idx]) for word, idx in countvec.vocabulary_.items())
    words_freq = pd.Series(words_freq).sort_values(ascending=False)
    return words_freq
