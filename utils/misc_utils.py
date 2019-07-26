import matplotlib
import matplotlib.pyplot as plt
import squarify
import pandas as pd
from utils.config import config as cf


def clean_data(df):
    """
    Strip and lower String columns
    :param df: pd.DataFrame
    :return: cleaned pd.DataFrame
    """
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.lower().str.strip()
    return df


def load_categories(filename=cf.CATEGORY_FILE):
    """
    Load category from file
    :param filename:
    :return: pd.DataFrame of ['Type', 'Main Category', 'Category']
    """
    print('======= Load categories ::: ', filename)
    cat = pd.read_excel(filename)
    print('Total ', len(cat), ' rows')
    cat = cat[['Type', 'Main Category', 'Category']]
    cat = clean_data(cat)
    return cat


def load_suburb_names(filename=cf.SUBURBS_FILE):
    """
    Load suburb names from file
    :param filename: Input file name
    :return: pd.DataFrame, Return only suburb names which are >= 3 characters
    """
    print('======= Load Cities name ::: ', filename)
    suburb = pd.read_csv(filename, header=None)
    print('Total ', len(suburb), ' rows')
    suburb = clean_data(suburb)
    # sort by longest length for longest match replace
    suburb['len'] = suburb[0].str.len()
    suburb = suburb.drop(suburb[suburb['len'] <=3].index)
    suburb = suburb.sort_values('len', ascending=False)
    return suburb


def load_stopwords(filename=cf.STOPWORDS_FILE):
    """
    Load Custom stopwords from file
    :param filename: Input file name
    :return: pd.DataFrame of stopwords
    """
    print('======= Load Stopword name ::: ', filename)
    stopwords = pd.read_csv(filename, header=None)
    print('Total ', len(stopwords), ' rows')
    stopwords = clean_data(stopwords)
    return stopwords


def load_currencies(filename=cf.CURRENCIES_FILE):
    """
    Load currencies names from file
    Raw file from :https://raw.githubusercontent.com/umpirsky/currency-list/master/data/en_GB/currency.csv
    :param filename: Input file name
    :return: pd.DataFrame, [id, value]
    """
    print('======= Load Currency ::: ', filename)
    currencies = pd.read_csv(filename)
    print('Total ', len(currencies), ' rows')
    currencies = clean_data(currencies)
    return currencies


def load_country_code(filename=cf.COUNTRY_CODE_FILE):
    """
    Load country codes from file
    Data from :https://www.iban.com/country-codes
    :param filename: Input file name
    :return: pd.DataFrame, [Country,Alpha-2,Alpha-3]
    """
    print('======= Load Country code ::: ', filename)
    currencies = pd.read_csv(filename, na_filter=False)
    print('Total ', len(currencies), ' rows')
    currencies = clean_data(currencies)
    return currencies


def process_suburb_list():
    """Create suburb list from nzpostcodes_v2.csv
        the output file is saved in 'resources/nz_cities.csv'
    """
    cities = pd.read_csv(cf.SUBURBS_RAW_FILE)
    # Clean data by removing '.Airport, .Central,.North,....'
    cities['suburb'] = cities['suburb'].str.lower()
    remove_list = [' airport', ' central', ' north', ' south', ' east', ' west', ' bay', ' valley', ' beach', ' park']
    remove_list = '|'.join(remove_list)
    print('Remove list: ', remove_list)
    cities['suburb'] = cities['suburb'].str.replace(remove_list, '')

    cities = pd.DataFrame(cities.suburb.unique(), columns=['suburb'])
    cities = cities.sort_values(by='suburb').dropna()
    print('Total suburb : ', len(cities))

    cities.to_csv(cf.ROOT_PATH + 'resources/nz_cities.csv', index=False, header=False)
    print('export done...')


def write_excel(df_map, filename):
    """
    Export data frames to Excel sheets
    The file is exported to path cf.EXPORT_PATH
    :param df_map: A dictionary of {'sheet_name': DataFrame_to_export}
    :param filename: Output file name
    :return: None
    """
    with pd.ExcelWriter(cf.EXPORT_PATH + filename) as writer:  # doctest: +SKIP
        for name, df in df_map.items():
            df.to_excel(writer, sheet_name=name)
    print(filename + ' exported!')


def plot_tree_map(data, title=None, fig_size=(20, 15), cmap_name=None, filename=None):
    """
    :param data: pd.Series of data
    :param title: Chart title name
    :param fig_size: figure size (default = (20,15)
    :param cmap_name: Specify a custom color map. Default=None
    :param filename: Filename for export chart. If 'None'(Default), no file is exported.
    :return: None
    """

    fig = plt.gcf()
    fig.set_size_inches(fig_size)
    # color scale on the population
    if cmap_name is not None:
        cmap = plt.get_cmap(cmap_name)
        mini, maxi = data.min(), data.max()
        norm = matplotlib.colors.Normalize(vmin=mini, vmax=maxi)
        colors = [cmap(norm(value)) for value in data]
        # colors[1] = "#FBFCFE"
    else:
        colors = None

    squarify.plot(sizes=data, label=data.index, alpha=.8, color=colors)
    plt.title(title)
    plt.axis('off')

    if filename is not None:
        plt.savefig(cf.EXPORT_PATH + filename, bbox_inches='tight')
        print(filename + ' exported!')

    plt.show()


def test_utils():
    print('Test utils')
    categories = load_categories()
    suburbs = load_suburb_names()
    stopwords = load_stopwords()

    print(categories.head())
    print(suburbs.head())
    print(stopwords.head())
    return 1
