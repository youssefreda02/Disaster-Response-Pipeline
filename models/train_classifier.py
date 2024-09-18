import sys
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix,classification_report
import nltk
nltk.download(['punkt', 'wordnet','stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def load_data(database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('message_category', engine)
    # Define X and Y

    x_columns = ['message']
    X = df[x_columns]
    Y = df.drop(x_columns, axis=1)
    Y = Y.drop(['id', 'original', 'genre'], axis=1)
    #There was 195 cell that contain value of 2.0, however, droping them or replacing to be 1.0 is not a big deal.
    Y.replace(inplace=True, value=1.0, to_replace= 2.0)
    # Drop rows in Y with NaN values as the NAN rows about 140 rows. Small ratio
    Y.dropna(inplace=True)
    # Drop corresponding rows in X
    X = X.loc[Y.index]  # This keeps only the rows in X that correspond to the remaining rows in Y


    return X, Y, Y.columns


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf',MultiOutputClassifier(RandomForestClassifier()))
])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    # Predict on test data
    y_pred = model.predict(X_test['message'])
    
    # Print the overall accuracy
    print('Overall Accuracy: ', (y_pred == Y_test).mean().mean())

    # Loop over each category to generate a classification report
    for i, col in enumerate(category_names):
        print(f'Category: {col}\n')
        print(classification_report(Y_test[col], y_pred[:, i]))
        print('-' * 60)

def save_model(model, model_filepath):
    # Specify the path where you want to save the model
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train['message'], Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()