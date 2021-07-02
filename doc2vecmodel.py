import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from gensim.parsing.preprocessing import remove_stopwords
import re
import nltk
from sklearn import utils


nltk.download('stopwords')

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

def clean_text(text):
    '''Text Preprocessing '''
    #text = remove_stopwords(text)
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    return text

def clean(df):
    return df['text'].apply(clean_text)

def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens

tweet_data = pd.read_csv('data/train.csv')

test_data = pd.read_csv('data/test.csv')

#train, test = train_test_split(tweet_data, test_size=0.3, random_state=42)

train = tweet_data
test = test_data

train_tagged = train.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['text']), tags=[r.id]), axis=1)
test_tagged = test.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['text']), tags=[r.id]), axis=1)

import multiprocessing
cores = multiprocessing.cpu_count()

model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=2, workers=cores)
model_dbow.build_vocab([x for x in tqdm(train_tagged.values)])

for epoch in range(30):
    model_dbow.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)
    model_dbow.alpha -= 0.002
    model_dbow.min_alpha = model_dbow.alpha

def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors

y_1, X_train = vec_for_learning(model_dbow, train_tagged)
y_train = tweet_data['target']

y_2, X_test = vec_for_learning(model_dbow, test_tagged)

print(X_train[30])
logreg = LogisticRegression(C=1e5)
clf = logreg.fit(X_train, y_train)
y_test = clf.predict(X_test)

results_df = pd.DataFrame(y_test, columns = ['target'])

#Results formatted for Kaggle submission:
kaggle_submit = pd.concat([test_data['id'], results_df], axis = 1)
kaggle_submit.to_csv('results/kaggle_submit.csv',index=False)

#Results formatted for testing by eye:
results = pd.concat([test_data['text'], results_df], axis = 1)
results.to_csv('results/check_this.csv')

# from sklearn.metrics import accuracy_score, f1_score
# print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
# print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))
