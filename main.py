import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re
import numpy as np

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
#print(vectorizer.get_feature_names())
#print(X.shape)
y = tweet_data['target']
clf = LogisticRegression().fit(X, y)

# This code block prints the features
#with highest absolute value coefficients
features = vectorizer.get_feature_names()
coefs = clf.coef_
sorted_ind = np.argsort(*clf.coef_)
print('Most likely to be non-disaster tweet:')
for i in range(20):
    print(features[sorted_ind[i]], coefs[0][sorted_ind[i]])
print('Most likely to be disaster tweet:')
for i in range(1,21):
    print(features[sorted_ind[-i]], coefs[0][sorted_ind[-i]])
#########

test_data = pd.read_csv('data/test.csv')
X_test = vectorizer.transform(clean(test_data))
y_test = clf.predict(X_test)

results_df = pd.DataFrame(y_test, columns = ['target'])

#Results formatted for Kaggle submission:
kaggle_submit = pd.concat([test_data['id'], results_df], axis = 1)
kaggle_submit.to_csv('results/kaggle_submit.csv',index=False)

#Results formatted for human reading
results = pd.concat([test_data['text'], results_df], axis = 1)
results.to_csv('results/check_this.csv')
