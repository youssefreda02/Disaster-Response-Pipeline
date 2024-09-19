import sys
import pandas as pd
from sqlalchemy import  create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load and merge messages and categories datasets.

    Args:
    messages_filepath: str. Path to the messages CSV file.
    categories_filepath: str. Path to the categories CSV file.

    Returns:
    df: pandas DataFrame. Merged DataFrame containing both messages and categories data.
    """
    # Load messages dataset
    messages = pd.read_csv(messages_filepath)
                           
    # Load categories dataset
    categories = pd.read_csv(categories_filepath)
  
    # Merge the messages and categories datasets using the common 'id' column
    df = messages.merge(categories, on='id')
    
    return df


def clean_data(df):
    """
    Clean and preprocess the merged dataset.

    - Splits the categories column into separate columns.
    - Converts category values to binary (0 or 1).
    - Removes duplicates from the dataset.

    Args:
    df: pandas DataFrame. The merged DataFrame containing both messages and categories.

    Returns:
    df: pandas DataFrame. The cleaned DataFrame with split categories and no duplicates.
    """
    # Split the values in the 'categories' column on the ';' character
    # so that each value becomes a separate column.
    row = df['categories'][1].split(';')
    
    # Extract category column names by removing the last two characters from each category value
    category_colnames = [x[0:-2] for x in row]
    
    # Preserve original column names for later use
    old_columns = df.columns
    
    # Expand categories into separate columns and merge with the original DataFrame
    df = df.merge(df['categories'].str.split(';', expand=True), left_index=True, right_index=True)
    
    # Update the DataFrame's columns with the new category names
    df.columns = pd.Index(old_columns.tolist() + category_colnames)
    
    # Drop the original 'categories' column
    df.drop(['categories'], axis=1, inplace=True)

    # Convert category values to numeric (0 or 1) for each category column
    for column in category_colnames:
        # Set each value to the last character of the string (0 or 1)
        df[column] = df[column].str[-1]
        # Convert the column from string to numeric
        df[column] = pd.to_numeric(df[column])

    # Drop duplicates from the DataFrame
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """
    Save the cleaned data to a SQLite database.

    Args:
    df: pandas DataFrame. The cleaned data.
    database_filename: str. The filename for the SQLite database.

    Returns:
    None
    """
    # Create SQLite engine
    engine = create_engine(f'sqlite:///{database_filename}')
    
    # Save DataFrame to the SQLite database, replacing existing data if necessary
    df.to_sql('message_category', engine, index=False, if_exists='replace')
    
    pass


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
