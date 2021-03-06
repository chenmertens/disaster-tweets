import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
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
y = tweet_data['target']

# This code block splits a validation test set
# and calculates the model performance on it.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
print (list(map(lambda x: x.shape, [X_train, X_test, y_train, y_test])))

clf = LogisticRegression().fit(X_train, y_train)

y_hat = clf.predict(X_test)
print (np.mean((y_hat - y_test)**2))
print ((sum(y_hat == y_test)) / len(y_hat))

##############################################################

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
# print(results_df.describe())

#Results formatted for Kaggle submission:
kaggle_submit = pd.concat([test_data['id'], results_df], axis = 1)
kaggle_submit.to_csv('results/kaggle_submit.csv',index=False)

#Results formatted for human reading
results = pd.concat([test_data['text'], results_df], axis = 1)
results.to_csv('results/check_this.csv')
