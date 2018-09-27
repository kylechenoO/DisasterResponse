import re
import sys
import nltk
import pickle
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

def load_data(database_filepath):
    '''
        load_data is a function to load data from dbfile
        input: database_filepath
        output: X, Y, Y_labels
    '''

    ## load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('InsertTableName', engine)

    Y_labels = ['related', 'request', 'offer', 'aid_related', 
        'medical_help', 'medical_products', 'search_and_rescue', 
        'security', 'military', 'child_alone', 'water', 'food', 
        'shelter', 'clothing', 'money', 'missing_people', 'refugees', 
        'death', 'other_aid', 'infrastructure_related', 'transport', 
        'buildings', 'electricity', 'tools', 'hospitals', 'shops', 
        'aid_centers', 'other_infrastructure', 'weather_related', 
        'floods', 'storm', 'fire', 'earthquake', 'cold', 
        'other_weather', 'direct_report']
    X = df['message'].values
    Y = df[Y_labels].values

    ## return val
    return(X, Y, Y_labels)

def tokenize(text):
    '''
        tokenize is a function to process text data
        input: text
        output: clean tokens
    '''

    ## load stop words and wordnetlemmatizer
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    
    ## normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    ## tokenize text
    tokens = word_tokenize(text)
    
    ## lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    ## return val
    return(tokens)

def build_model():
    '''
        build_model is a function to build model
        input: no input values
        output: model
    '''

    ## build pipline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(random_state = 42), n_jobs = -1))
    ])

    ## return val
    return(pipeline)

def evaluate_model(model, X_test, Y_test, category_names):
    '''
        evaluate_model is a function to evaluate model
        input: model, X_test, Y_test, category_names
        output: no return values, just print f1-score of X_test
    '''

    ## get the f1 score of X_test
    y_pred = model.predict(X_test)
    f1_list = []
    for t, p in zip(Y_test, y_pred):
        f1_list.append(f1_score(t, p, average = 'macro'))

    print(np.array(f1_list).mean())

def save_model(model, model_filepath):
    '''
        save_model is a function to save final model
        input: model, model_filepath
        output: no return values, just save file
    '''

    ## save model
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
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
