import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Load data from csv
    args: 
        messages_filepath: path to the messages.csv file
        categories_filepath: path to  the categories.csv file
    return:
        merged dataframe of message.csv and categories.csv
    """
    #Loading data from csv file
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    #merging data
    df = pd.merge(messages, categories, how='outer', on='id')
    
    return df

def clean_data(df):
    """
    1. Create different columns for different categories
    2. Handle the duplicate values
    3. Merge those new columns to main dataframe
    4. Handle if there's any NaN value
 
    args:
        merged dataframe from load data
    return:
        Clean data
    """
    categories = df["categories"].str.split(";", expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x.split('-')[0])
    #creating columns
    categories.columns = category_colnames
    
    
    #converting categorical value to the numerical
    for column in categories:
        categories[column] = categories[column].apply(lambda x: int(x.split("-")[1]))
    
    #dropping the original categories column from `df`
    df.drop(['categories'],axis=1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    #check number of duplicates and eliminate it if find some
    if sum(df.duplicated())>0:
        df = df.drop_duplicates()
        
    #return clean data
    return df


def save_data(df, database_filename):
    """Save the clean data to the database
    
    args:
        df: clean data
        database_filename: Name of the database in which data needs to be saved
        
    returns:
        None
    """  
    engine = create_engine('sqlite:///DisasterResponse.db')
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')
    


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