class config :

    # PATH CONFIG
    ROOT_PATH = '/project/dissertation/'
    EXPORT_PATH = ROOT_PATH + 'output/'
    DATA_PATH = ROOT_PATH + 'Transaction Data/'
    RESOURCES_PATH = ROOT_PATH + 'resources/'

    # FILE NAME
    CATEGORY_FILE = DATA_PATH + 'Category_new.xlsx'
    SUBURBS_FILE = RESOURCES_PATH + 'nz_cities.csv'
    CURRENCIES_FILE = RESOURCES_PATH + 'currency.csv'
    COUNTRY_CODE_FILE = RESOURCES_PATH + 'country_code.csv'
    STOPWORDS_FILE = RESOURCES_PATH + 'stop_words.csv'
    SUBURBS_RAW_FILE = RESOURCES_PATH + 'nzpostcodes_v2.csv'
    
    #Pre-trained Word-Embeddings
    PRETRAINED_W2V_BIN = '/project/GoogleNews-vectors-negative300.bin'
    PRETRAINED_FASTTEXT_BIN = '/project/wiki-news-300d-1M-subword.bin'
    ### GLOVE pickled files which have been pre-processed with '/utils/load_glove.py'
    GLOVE_6B_50D_P_FILE = RESOURCES_PATH + 'GLOVE_6B_50D.p'
    GLOVE_6B_100D_P_FILE = RESOURCES_PATH + 'GLOVE_6B_100D.p'
    GLOVE_6B_300D_P_FILE = RESOURCES_PATH + 'GLOVE_6B_300D.p'
    GLOVE_42B_300D_P_FILE = RESOURCES_PATH + 'GLOVE_42B_300D.p'
    GLOVE_840B_300D_P_FILE = RESOURCES_PATH + 'GLOVE_840B_300D.p'

    DATA_FILES = {
                  'file1': 'file1.xlsx'
                  }

    # GLOBAL CONSTANTS
    RANDOM_ST = 99
    CV = 5
    BASE_COUNTRY_CODE_ALPHA3 = 'nzl'
    BASE_COUNTRY_CODE_ALPHA2 = 'nz'
