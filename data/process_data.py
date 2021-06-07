# import libraries
import pandas as pd
from sqlalchemy import create_engine
import sys


def load_data(messages_filepath, categories_filepath):
    """ Reads in the messages and categories data sets and merges them into a single dataframe.
    Args:
        messages_filepath: String of file path to the messages data.
        categories_filepath: String of file path to the categories data.
    Returns:
        Returns a merged dataframe of both inputs.
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on="id")

    return df


def clean_data(df):
    """Cleans categorical target columns.
    Args:
        df:  Dataframe result from load_data() function.
    Returns:
        A tidy dataframe with cleaned up categorical columns.
    """
    categories = df["categories"].str.split(";", expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = [value[:-2] for value in row]

    # rename the columns of categories
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df = df.drop("categories", axis = 1)
    df = pd.concat([df, categories], axis = 1)

    # Clean up values greater than 1:
    for col in df.columns[4:]:
        if df[col].max() > 1:
            df.loc[df[col] > 1, col] = 1

    # drop duplicates
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    """Save tidy dataframe to sqlite database.
    Args:
        df: tidy dataframe generated from clean_data().
        database_filename: Name of sqlite database.
    Returns:
        None
    """

    engine = create_engine('sqlite:///./{}'.format(database_filename))
    df.to_sql('messages', engine, index=False, if_exists="replace")
    return None


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