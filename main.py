import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re

vectorizer = TfidfVectorizer()

tweet_data = pd.read_csv('data/train.csv')
print(tweet_data.describe())

def clean_text(text):
    '''Text Preprocessing '''

    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    return text

def clean(df):
    return df['text'].apply(lambda stringliteral: clean_text(stringliteral))

X = vectorizer.fit_transform(clean(tweet_data))
print(vectorizer.get_feature_names())
print(X.shape)
y = tweet_data['target']
clf = LogisticRegression().fit(X, y)

test_data = pd.read_csv('data/test.csv')
X_test = vectorizer.transform(clean(test_data))
y_test = clf.predict(X_test)

results_df = pd.DataFrame(y_test)
print(results_df.describe())

results_df.to_csv('results/results.csv')

results = pd.concat([test_data['text'], results_df], axis = 1)
results.to_csv('results/check_this.csv')
