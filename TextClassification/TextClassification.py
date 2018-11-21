import sys
import os
import pandas as pd
from bs4 import BeautifulSoup as bs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC

from sklearn import metrics
from pyvi import ViTokenizer

class TopicClassification(object):
    def __init__(self, training_data):
        self.training_data = training_data

    def get_data(self):
        dict_store = []
        f = open(self.training_data, 'r')
        f_bs = bs(f, 'html.parser')
        documents = f_bs.find_all('document')
        for document in documents:
            soup_doc = bs(str(document), 'html.parser')
            dict_store.append({'label': soup_doc.label.get_text(), 'content': soup_doc.content.get_text()})
        return dict_store

def main():
    if len(sys.argv) != 2:
        print("TopicClassification.py training_data")
    else:
        training_data = sys.argv[1]
        # tc: topic_classification

        tc = TopicClassification(training_data)
        dict_store = tc.get_data()
        df = pd.DataFrame(dict_store)

        col = ['label', 'content']
        df = df[col]
        df = df[pd.notnull(df['content'])]

        df.columns = ['label', 'content']
        df['label_id'] = df['label'].factorize()[0]
        label_id_df = df[['label', 'label_id']].drop_duplicates().sort_values('label_id')
        label_to_id = dict(label_id_df.values)
        id_to_label = dict(label_id_df[['label_id', 'label']].values)
        tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='utf8', ngram_range=(1, 2))

        features = tfidf.fit_transform(df.content).toarray()
        labels = df.label_id

        model = LinearSVC()
        X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.33, random_state=0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        f = open('result.txt', 'w')
        print(metrics.classification_report(y_test, y_pred, target_names=df['label'].unique()))
        #f.write(metrics.classification_report(y_test, y_pred, target_names=df['label'].unique()))

if __name__ == "__main__":
    main()
