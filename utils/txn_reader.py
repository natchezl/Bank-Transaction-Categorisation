import pandas as pd
import numpy as np
from utils import misc_utils
from utils.config import config as cf
from sklearn.preprocessing import StandardScaler


def load_all_transaction_files():
    """
    Load all transaction from the data_files variable
    :return: pd.DataFrame
    """
    df_list = []
    for (key, file_name) in cf.DATA_FILES.items():
        df = read_txn_xlsfile(cf.DATA_PATH + file_name)
        df_list.append(df)
    all_df = pd.concat(df_list, axis=0, ignore_index=True, sort=False)
    all_df = preprocess_transaction(all_df)
    print('Load file completed! Total : ', len(all_df), ' transactions')
    return all_df


def filter_min_count_category(df, min_count=6):
    """
    Filter dataframe where category > min_count
    :param df: (pd.DataFrame) : transaction dataframe
    :param min_count: (int) minimum number of transaction in the category
    :return: filtered dataframe
    """
    data_by_cat = pd.DataFrame(df.groupby(['Category', 'Main Category', 'Type']).size(), columns=['count'])
    data_by_cat = data_by_cat.sort_values('count', ascending=False)
    df = df.join(data_by_cat, on=['Category', 'Main Category', 'Type'], how='left')
    df = df.loc[df['count'] >= min_count].reset_index()
    df = df.drop(columns=['count'])
    print('filtered dataframe : ', len(df))
    # #backup description
    # df['desc'] = df['Description']
    return df


def read_txn_xlsfile(filename):
    """
    Load transaction from Excel file
    and pre-processed by bank name & account type
    :param filename:
    :return: pd.DataFrame
    """
    print('======= File ::: ', filename)
    df = pd.read_excel(filename, parse_dates=['Date'])
    print('Total ', len(df), ' rows')
    # print(df.info())

    df = preprocess_by_bank(df)
    return df


def preprocess_by_bank(df):
    """
    Pre-process transaction data by bank name & account type

    :param df:
    :return: pd.DataFrame ['Date', 'Description', 'Amount', 'Category', 'Business/Personal','Bank', 'Account Type']
    """
    bank_name = df['Bank'][0].lower()
    acct_type = df['Account Type'][0].lower()
    print('Bank name: ', bank_name)
    print('Acct type: ', acct_type)
    df['Bank'] = df['Bank'].fillna(method='ffill')
    df['Account Type'] = df['Account Type'].fillna(method='ffill')
    df['isOverseas'] = 0

    if bank_name == 'kiwibank':
        df = preprocess_kiwibank(df)

    elif bank_name == 'tsb':
        df = preprocess_tsb(df, acct_type)

    elif bank_name == 'westpac':
        df = preprocess_westpac(df)

    elif bank_name == 'anz':
        df = preprocess_anz(df, acct_type)

    else:
        raise ValueError('Unknown bank name - ', bank_name)

    return df


def preprocess_kiwibank(df):
    df = filter_columns(df)
    # Remove text after ';'
    df['Description'] = df['Description'].str.split(';').str[0]
    return df


def preprocess_tsb(df, acct_type):
    if acct_type == 'credit card':
        df['Description'] = df['Description'] + df['Particulars'].astype(str)
        # Extract the last 2 Characters from the description and remove it from the description field
        # eg. RELAY BILINGA AU (115) -> country = 'AU', Description = RELAY BILINGA (115)
        df['country'] = df['Description'].str.extract(r' ([A-Za-z]{2})( *$| \(| *-\d)')[0]
        df['Description'] = df['Description'].replace({r' ([A-Za-z]{2})( *$| \(| *-\d)': r' \2'}, regex=True)
        # If the country code != BASE_COUNTRY_CODE_ALPHA3, then isOverseas = True
        df['isOverseas'] = (df['country'].str.lower() != cf.BASE_COUNTRY_CODE_ALPHA2).astype('int')

    elif acct_type == 'everyday':
        df['Particulars'] = df['Particulars'].fillna('')
        df['Description'] = df['Description'].str.replace(pat='^T/F', repl='transfers', case=False, regex=True)
        df['Description'] = df[['Description', 'Particulars']].apply(check_description_particulars, axis=1)
    else:
        raise ValueError('Unknown account type - ', acct_type)
    df = filter_columns(df)
    return df


def preprocess_westpac(df):
    """
    Note: Transaction from Westpac contains multiple account types in a single file
    so we need to process transaction line by line
    :param df:
    :return:
    """
    df['Other party'] = df['Other party'].fillna('')
    df = df.apply(preprocess_westpac_description, axis=1)
    df = df.drop(columns=['Description'])
    df = df.rename(columns={'Other party': 'Description'})
    df = filter_columns(df)
    return df


def preprocess_westpac_description(row):
    acct_type = row['Account Type']
    if len(row['Other party']) == 0:
        row['Other party'] = row['Description']

    if acct_type.lower() == 'credit card':
        # Split city and country from the description by fixed length
        # eg. THE CHURCHILL          WELLINGTON    NZL
        desc = row['Other party'][:23].strip()
        # city = row['Description'][23:37].strip()
        country = row['Other party'][37:].strip()
        row['Other party'] = desc
        row['isOverseas'] = 0 if country.lower() == cf.BASE_COUNTRY_CODE_ALPHA3 else 1
    return row


def preprocess_anz(df, acct_type):
    if acct_type == 'credit card':
        df['Amount_sign'] = np.where(df['Type'] == 'D', -1, 1)
        df['Amount'] = df['Amount'] * df['Amount_sign']
    else:
        df['Description'] = df['Description'].fillna(df['Type'])
    df = filter_columns(df)
    return df


def filter_columns(df):
    return df[['Date', 'Description', 'Amount', 'Category', 'Business/Personal', 'Bank', 'Account Type', 'isOverseas']]

# def preprocess_anz_description(row):
#     """
#     If the D
#     :param row:
#     :return:
#     """
#     if row['Description'] == np.nan:
#         row['Description'] = row['Type']
#     return row


def check_description_particulars(row):
    """If the last char of description field is 'space',
    then append Particulars column to the description
    """
    if (len(str(row['Description'])) == 24) and (str(row['Description'])[-1] != ' '):
        row['Description'] = str(row['Description']) + str(row['Particulars'])
    return row


def preprocess_transaction(df):
    """
    Pre-processing transaction data
    - Fill N/A for Business/Personal column
    - Drop rows that don't have Category
    - Extract 'isExpense' column from Amount
    - Create 'Amount_logabs' = Transformed Log-abs amount
    - Create 'Type' column (Expense/Income)
    - Map Txn with its Main-Category
    - Create 'Label' column = '<Type>::<Main Category>::<Category>'
    :param df:
    :return: pd.DataFrame
    """
    # check null values
    df['Description'] = df['Description'].fillna('-')
    df['Business/Personal'] = df['Business/Personal'].fillna('Personal')
    # drop rows which don't have label
    df.dropna(subset=['Category'], inplace=True)
    print('Total rows ', len(df))

    df['isExpense'] = (df['Amount'] <= 0).astype('int')
    # Transform Log absolute amount
    df['Amount_logabs'] = np.log(df['Amount'].abs()+0.01).values.reshape(-1,1)
    
    df['Type'] = df['Amount'].apply(lambda x: 'Expenses' if x <= 0 else 'Income')
    # df['Description'] = df['Description']+ ' '+ df['isExpense'].map(str)

    df = misc_utils.clean_data(df)
    # Merge with categories file to get Main-Categories
    categories = misc_utils.load_categories()
    df = df.merge(categories, on=['Category', 'Type'], how='inner')

    # Create label column
    df['label'] = (df['Type'] + '::' + df['Main Category'] + '::' + df['Category']).str.title()

    return df

# all_data = load_all_transaction_files()
# print(all_data.head())
