import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score as score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import make_pipeline
import io
from collections import Counter
from scipy.sparse import hstack
import nltk
from nltk.stem.porter import PorterStemmer


data_train = pd.read_csv('E:\\PycharmProjects\\Machine_L\\Data_Science_Club\\second_\\train.csv')
data_test = pd.read_csv('E:\\PycharmProjects\\Machine_L\\Data_Science_Club\\second_\\test.csv')
class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

all_text = pd.concat([data_train['comment_text'], data_test['comment_text']])
vect_tf = TfidfVectorizer(min_df = 5, max_df = 40000)
train_text_features_tf = vect_tf.transform(data_train['comment_text'])
test_text_features_tf = vect_tf.transform(data_test['comment_text'])

grid = {'C': np.power(10.0, np.arange(-5, 5))}
scores_vectorizer_tf = []
best_params_vectorizer_tf = []
for cl_name in class_names:
    y_train = data_train[cl_name]
    gs = GridSearchCV(LogisticRegression(penalty = 'l2', solver='sag'), grid, n_jobs = -1, scoring = 'roc_auc', cv=KFold(n_splits = 5, shuffle = True, random_state = 241))
    gs.fit(train_text_features_tf, y_train)
    scores_vectorizer_tf.append(gs.best_score_)
    best_params_vectorizer_tf.append(gs.best_params_)
    for scores in gs.grid_scores_:
        print(scores.mean_validation_score)
        print(scores.parameters)

mean_score = np.mean(scores_vectorizer_tf)# среднее значение

submission = pd.DataFrame.from_dict({'id': data_test['id']})
for cl_name, C_par in zip(class_names, best_params_vectorizer_tf):
    clf = LogisticRegression(C = C_par, solver='sag')
    y_train = data_train[cl_name]
    clf.fit(train_text_features_tf, y_train)
    submission[cl_name] = clf.predict_proba(test_text_features_tf)[:, 1]
submission.to_csv('submission_first.csv', index=False)