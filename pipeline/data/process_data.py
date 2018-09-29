import re
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
        load_data is a function to load and preprocess data from orgi csv file
        input: messages_filepath, categories_filepath
        output: df after preprocessing
    '''

    ## load from csv
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    ## merge to df
    df = pd.merge(messages, categories, left_on = 'id', right_on = 'id')

    ## create a dataframe of the 36 individual category columns
    categories = categories['categories'].str.split(";", expand=True)

    ## select the first row of the categories dataframe
    row = categories.loc[0]

    ## use this row to extract a list of new column names for categories.
    ## one way is to apply a lambda function that takes everything 
    ## up to the second to last character of each string with slicing
    category_colnames = [ re.sub('-.*$', '', x) for x in row ]

    ## rename the columns of `categories`
    categories.columns = category_colnames

    ## Convert category values
    for column in categories:
        ## set each value to be the last character of the string
        categories[column] = categories[column].str.replace('^.*-', '')

        ## convert column from string to numeric
        categories[column] = categories[column].astype(int)

    ## drop the original categories column from `df`
    df.drop(['categories'], axis = 1, inplace = True)

    ## concatenate the original dataframe with the new `categories` dataframe
    df = pd.merge(df, categories, left_on = df.index.values, right_on = categories.index.values)
    df.drop(['key_0'], axis = 1, inplace = True)

    ## return df
    return(df)

def clean_data(df):
    '''
        clean_data is a function to clean the data from df
        input: df
        output: df after clean
    '''

    ## Remove duplicates
    df = df.drop_duplicates()

    ## return df
    return(df)


def save_data(df, database_filename):
    '''
        save_data is a function to safe df to dbfile
        input: df
        output: no return val, but saved dbfile
    '''

    ## Save the clean dataset into an sqlite database
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('InsertTableName', engine, index = False)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
