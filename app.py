from flask import Flask, jsonify
import requests

app = Flask(__name__)

import nltk
import collections
from wordcloud import WordCloud, STOPWORDS
import pandas as pd
import itertools
import re 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import string
from textblob import TextBlob
from nltk.corpus import stopwords
analyzer = SentimentIntensityAnalyzer()

# Save .csv to dataframe and clean the data 

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

# Cleans the df into ['text'] and stuff.
def clean_df(df, keep_punctuation):
    # Drop duplicate tweets
    df = df.drop_duplicates()
        
    # Removing numbers/numerics 
    def cleaning_numbers(text):
        return re.sub('[0-9]+', '', text)
    df['text'] = df['text'].apply(lambda x: cleaning_numbers(x))
    df['text'].tail()

    # Removing punctuation 
    english_punctuations = string.punctuation
    punctuations_list = english_punctuations
        
    if(keep_punctuation):
        def cleaning_punctuations(text):
            translator = str.maketrans('', '', punctuations_list)
            return text.translate(translator)
        df['text']= df['text'].apply(lambda x: cleaning_punctuations(x))
        df['text'].tail()

    # Removing URLs 
    def cleaning_URLs(text):
        return re.sub('((www.[^s]+)|(https?://[^s]+))',' ',text)
    df['text'] = df['text'].apply(lambda x: cleaning_URLs(x))
    df['text'].tail()

    # Convert everything to lowercase 
    df['text']= df['text'].str.lower()
    df['text'].tail()
    return df
    
needed_stop = ['against', 'above', 'below', "haven't", "won't", "mightn't", "not", "needn't", "wouldn't", "shan'", "weren't", "didn't", "hadn't", "wasn't", "don'", "didn'", "didn't", "don",
            'couldn', 'didn', 'doesn', "doesn't", 'hadn', 'hasn', "hasn't", 'haven', 'isn', 'ma', 'mightn', 'mustn', "mustn't", 'needn', 'shan', 'shouldn', "shouldn't", 'wasn', 'weren', 'won', 'wouldn',
            "don't", "mustn", "shan", "ain", "aren",  "isn't", "no", "isn'", "shan't", "aren't", "couldn't", "mustn’t", "shouldn’t", "shouldn’", "aren’", "ain’", "ain’t", "couldn’", "needn’t", "wasn’", "weren’", "won’", "won’t", "wouldn’"] 

# Returns updated df, and the updated list of stopwords.
def clean_stopwords_except_needed(df, context_aware_stopwords):
    stop_words = stopwords.words('english')
    print(stop_words)
    # Remove some stopwords but do not include negative stopwords for more accurate polarity scores 
    stop_words = stop_words + context_aware_stopwords

    stop_words = list(filter(lambda i: i not in needed_stop, stop_words))

    def cleaning_stopwords(text):
        return " ".join([word for word in str(text).split() if word not in stop_words])
    df['text'] = df['text'].apply(lambda text: cleaning_stopwords(text))
    return df, stop_words

# Only returns the DF after cleaning all stop words provided, AND needed_stop.
def clean_stopwords_all(df, stop_words):
    # Remove all stopwords for better word frequency charting
    stop_words2 = list(set(stop_words + needed_stop))

    def cleaning_stopwords2(text):
        return " ".join([word for word in str(text).split() if word not in stop_words2])
    df['text'] = df['text'].apply(lambda text: cleaning_stopwords2(text))
    return df

# Outputs a word counter
def create_word_counter(df): 
    # Removing punctuation 
    english_punctuations = string.punctuation
    punctuations_list = english_punctuations
        
    def cleaning_punctuations(text):
        translator = str.maketrans('', '', punctuations_list)
        return text.translate(translator)
    df['text']= df['text'].apply(lambda x: cleaning_punctuations(x))
    df['text'].tail()
        
    all_tweets = df['text']
    words_in_tweet = [tweet.lower().split() for tweet in all_tweets]
    all_words = list(itertools.chain(*words_in_tweet))

    words = ",".join(str(item) for innerlist in df['text'].str.split() for item in innerlist).split(",")

    no_urls_no_tags = " ".join([word for word in words
                                if 'http' not in word
                                    and '@' not in word
                                ])

    counts_words = collections.Counter(all_words)

    words_tweets = pd.DataFrame(counts_words.most_common(40),
                                columns=['words', 'count'])
    return words_tweets

#############################################################################################
# Sentiment Analysis - returns an array of [positive_articles, neutral_articles, negative_articles]
def generate_sentiment_polarity_vader(df): 
    df2 = []
    for tweet in df['text']:
        ps = analyzer.polarity_scores(tweet)
        df2.append({'text':tweet, 'score':ps['compound']})

    tweetdf = pd.DataFrame(df2)

    positive_article = []
    neutral_article = []
    negative_article = []

    # Create list of polarity valuesx and tweet text
    sentiment_values = []

    # Conduct sentiment analysis on each tweet
    for ps in tweetdf.score:
        if ps > 0:
            positive_article.append(ps)
        elif ps == 0:
            neutral_article.append(ps)
        else:
            negative_article.append(ps)
    return [positive_article, neutral_article, negative_article]

# Sentiment Analysis - returns an array of [positive_articles, neutral_articles, negative_articles]
def generate_sentiment_analysis_textblob(df): 
    # sentiment_list = [TextBlob(tweet) for tweet in df['text']]
    # sentiment_values = [[tweet.sentiment.polarity, str(tweet)] for tweet in sentiment_list]
    # sentiment_df = pd.DataFrame(sentiment_values, columns=["polarity", "tweet"])

    df2 = []
    for tweet in df['text']:
        # Step 4 Perform Sentiment Analysis on Tweets
        df2.append({'text':tweet, 'score':TextBlob(tweet).sentiment.polarity})

    tweetdf = pd.DataFrame(df2)

    positive_article = []
    neutral_article = []
    negative_article = []

    # Create list of polarity valuesx and tweet text
    sentiment_values = []

    # Conduct sentiment analysis on each tweet
    for ps in tweetdf.score:
        if ps > 0:
            positive_article.append(ps)
        elif ps == 0:
            neutral_article.append(ps)
        else:
            negative_article.append(ps)
    return [positive_article, neutral_article, negative_article]

    return sentiment_df.values.tolist()

# Get 
@app.route('/', methods=['GET'])
def get_homepage():
    return "WORKING"

# Get 
@app.route('/data/<string:term>/<string:library>', methods=['GET'])
def get_data(term, library):
    df = pd.read_csv(term + ".csv", header=None)
    df.columns=['text']

    # Clean dataframe
    df = clean_df(df, False)
    # Sentiment analysis before stopword clearing

    custom_stopwords = []
    if(term == "abortion"):
        custom_stopwords = ['abortion', 'abortions', 'brt', 'the', 'one', 'i', 'us', 'booksnnthe', 'itxexxxexxa', 'platformxexxa', 'st', 'ajplus', 'f1', 'bc', 'st', "b'rt", "&amp;",  "b\"rt", "-", "b'with", "n.kore\\xe\\x\\xa", "lybia", ",", "abortion.", "don\\xe\\x\\xt", "it\\xe\\x\\xs", "would", "i\\xe\\x\\xm",  "\\xe\\x\\x", "b'the", "I", "i'm"]
    elif(term == "ukraine"):
        custom_stopwords = ['zelenskyy', "b'rt", "&amp;",  "b\"rt", "-", "b'with", "n.kore\\xe\\x\\xa", "lybia", "don\\xe\\x\\xt", "it\\xe\\x\\xs", "would", "i\\xe\\x\\xm",  "\\xe\\x\\x", "b'the", "I", "i'm"]
    
    polarity = [] 
    if(library == "vader"):
        polarity = generate_sentiment_polarity_vader(df)
        df, stop_words = clean_stopwords_except_needed(df, custom_stopwords)
        df = clean_stopwords_all(df, stop_words)

    elif(library == 'vader_stopwords'):
        df, stop_words = clean_stopwords_except_needed(df, custom_stopwords)
        polarity = generate_sentiment_polarity_vader(df)
        df = clean_stopwords_all(df, stop_words)
    
    elif(library == 'textblob'):
        polarity = generate_sentiment_analysis_textblob(df)
        df, stop_words = clean_stopwords_except_needed(df, custom_stopwords)
        df = clean_stopwords_all(df, stop_words)

    elif(library == 'textblob_stopwords'):
        df, stop_words = clean_stopwords_except_needed(df, custom_stopwords)
        polarity = generate_sentiment_analysis_textblob(df)
        df = clean_stopwords_all(df, stop_words)
    else:
        # default to using vader if nothing matched.
        polarity = generate_sentiment_polarity_vader(df)
        df, stop_words = clean_stopwords_except_needed(df, custom_stopwords)
        df = clean_stopwords_all(df, stop_words)

    # Create word counter 
    words_tweets = create_word_counter(df)
    return jsonify({'wordCounter': words_tweets.values.tolist(), 'polarity': polarity})

@app.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response


if __name__ == '__main__':
    app.run(host='localhost', port=9000, debug=True)
