import re
import json
import plotly
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def mytokenize(text):
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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('InsertTableName', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    ## genre_counts = df.groupby('genre').count()['message']
    ## genre_names = list(genre_counts.index)

    ## 1st plot data
    f_list = ['related', 'request', 'offer', 'aid_related', 
        'medical_help', 'medical_products', 'search_and_rescue', 
        'security', 'military', 'child_alone', 'water', 'food', 
        'shelter', 'clothing', 'money', 'missing_people', 'refugees', 
        'death', 'other_aid', 'infrastructure_related', 'transport', 
        'buildings', 'electricity', 'tools', 'hospitals', 'shops', 
        'aid_centers', 'other_infrastructure', 'weather_related', 
        'floods', 'storm', 'fire', 'earthquake', 'cold', 
        'other_weather', 'direct_report']
    sum_list = [ df[f].sum() for f in f_list ]

    ## 2nd plot data
    vect = CountVectorizer(tokenizer = mytokenize)
    X = vect.fit_transform(df[df['related'] == 1]['message'].values)
    voc_list = []
    count_list = []
    for key in vect.vocabulary_:
        voc_list.append(key)
        count_list.append(vect.vocabulary_[key])
    
    plt_voc = []
    plt_count = []
    base = 27900
    for i in range(0, 50):
        max_index = count_list.index(max(count_list))
        plt_count.append(count_list[max_index] - base)
        plt_voc.append(voc_list[max_index])
        del count_list[max_index]
        del voc_list[max_index]
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=f_list,
                    y=sum_list
                )
            ],

            'layout': {
                'title': 'Distribution of Every Class',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Class"
                }
            }
        },

        {
            'data': [
                Bar(
                    x=plt_voc,
                    y=plt_count
                )
            ],

            'layout': {
                'title': 'Voc Count of Related',
                'yaxis': {
                    'title': "Count(Need to add 27900)"
                },
                'xaxis': {
                    'title': "Voc"
                }
            }
        },
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    ####################################################################
    ## first plot end
    ####################################################################

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
