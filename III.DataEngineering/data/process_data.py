"""
Preprocessing Data for Disaster Response Pipeline (Udacity - Data Scientist Nanodegree)

Syntax
--------
    python process_data.py <messages_csv> <categories_csv> <output_db>

Example
----------
    python process_data.py disaster_messages.csv disaster_categories.csv DisasterResDB.db

Parameters
--------------
    messages_filepath (str) 
        path to messages.csv
    categories_filepath (str)
        path to categories.csv
    db_filepath (str)
        path to DisasterResDB.db
"""

import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load and merge messages and categories data from CSV files.

   Parameters
   --------------
        messages_filepath (str): 
            Path to the CSV file containing messages.
        categories_filepath (str): 
            Path to the CSV file containing categories.

    Returns
    ----------
        df (pandas DataFrame): 
            Merged DataFrame containing messages and categories.
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = pd.merge(
        messages, categories,
        on = "id"
    )
    print(f"Successfully merged {messages_filepath} with {categories_filepath}. Shape: {df.shape}")
    
    return df


def clean_data(df):
    """
    Clean and preprocess the DataFrame containing messages and categories.

    Parameters
    --------------
        df (pandas DataFrame): 
            DataFrame containing messages and categories.

    Returns
    ---------
        df (pandas DataFrame): 
            Cleaned and preprocessed DataFrame.
    """
    # create a dataframe of the 36 individual category columns
    categories = df["categories"].str.split(
        pat = ";", 
        expand = True
    )
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x : x[ : -2])
    categories.columns = category_colnames
    
    # Convert categories values into just 0 and 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    # drop the original categories column from `df`
    df = df.drop(
        columns = ["categories"]
    )
    
    df = pd.concat(
        objs = [df, categories],
        axis = 1
    )

    # Drop `child_alone` since it only contains 0, which is useless
    df = df.drop("child_alone", axis = 1)

    # Drop those rows that "related" == 2 since they might be just trashes or outliers
    df = df[
        df["related"] != 2
    ]

    # Drop duplicates, finally
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    """
    Save the cleaned DataFrame to a SQLite database.

    Parameters
    --------------
        df (pandas DataFrame):
            Cleaned and preprocessed DataFrame.
        database_filename (str):
            Path to the SQLite destination database.
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql(
        "disrestable", 
        engine, 
        index = False,
        if_exists = "replace"
    )


def main():
    """
    Main function to run the data processing script.

    Usage
    -------
        python process_data.py <path_to_messages_csv> <path_to_categories_csv> <path_to_output_db>

    Example
    ----------
        python process_data.py disaster_messages.csv disaster_categories.csv DisasterResDB.db
    """
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