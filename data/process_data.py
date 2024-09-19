import sys
import pandas as pd
from sqlalchemy import  create_engine


def load_data(messages_filepath, categories_filepath):
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
                           
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
  
    #Merge the messages and categories datasets using the common id
    
    df = messages.merge(categories, on = 'id')
    return df


def clean_data(df):
     #Split the values in the categories column on the ; character
     #so that each value becomes a separate column.
    row =  df['categories'][1].split(';')
    # Getting the columns names from the unique values of a row in the categories column
    
    category_colnames = [x[0:-2] for x in row]
    #Changing the columns names by adding unique values of the categories                       
    old_columns  = df.columns
    # Merging the new expanded columns to the orignal DataFrame
    df = df.merge(df['categories'].str.split(';', expand = True), left_index= True, right_index= True)       
    df.columns = pd.Index(old_columns.tolist() + category_colnames)
    
    #Dropping the old categories column
    df.drop(['categories'],axis = 1, inplace = True)

   
    #Setting the values of each row to be the last number it's value
    for column in df[category_colnames]:
        # set each value to be the last character of the string
        df[column] = df[column].str[-1]
        # convert column from string to numeric
        df[column] = pd.to_numeric(df[column])

  
    #Drop the duplicates
    df.drop_duplicates(inplace = True)

    return df


def save_data(df, database_filename):
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('message_category', engine, index=False)
    df.to_sql('message_category', engine, index=False, if_exists= 'replace')
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
