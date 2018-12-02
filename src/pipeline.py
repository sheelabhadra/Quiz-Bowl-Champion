import numpy as np
import pandas as pd
import yaml

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC

with open("config.yaml", 'r') as stream:
	cfg = yaml.load(stream)


class DataPreprocessing:
	def __init__(self, train_file, test_file):
		self.train_data = pd.read_csv(train_file)
		self.test_data = pd.read_csv(test_file)
		self.X = None
		self.y = None
		self.label_encoder = None

	def create_corpus(self, data):
		return [x['text'] for idx,x in data.iterrows()]

	def create_target(self, data):
		self.y = [x['cat'] for idx,x in data.iterrows()]
	
	def count_vectorize(self, corpus):
		vectorizer = CountVectorizer(stop_words='english')
		self.X = vectorizer.fit_transform(corpus)

	def tfidf_vectorize(self, corpus):
		vectorizer = TfidfVectorizer(stop_words='english')
		self.X = vectorizer.fit_transform(corpus)

	def reduce_tokens(self, keep_tokens=5000):
		count_tokens = np.sum(self.X, axis=0)
		indices = np.argsort(count_tokens)
		indices = indices.tolist()
		top_indices = indices[0][::-1]
		top_indices = top_indices[:keep_tokens]
		self.X = self.X[:, top_indices]

	def one_hot_encode(self):
		self.X[self.X > 0] = 1

	@property
	def y_train(self):
		self.label_encoder = preprocessing.LabelEncoder()
		return self.label_encoder.fit_transform(self.y)

	@property
	def X_train(self):
		_X_train = self.X[:len(self.train_data)]
		return _X_train

	@property
	def X_test(self):
		_X_test = self.X[len(self.train_data):]
		return _X_test

	def shuffle_train(self):
		X_train, y_train = shuffle(self.X_train, self.y_train, random_state=7)
		return X_train, y_train

	def __str__(self):
		return "===== TRAINING SET =====\n# observations: {}\n# features: {}".format(self.X_train.shape[0], self.X_train.shape[1])


class Classifier:
	def __init__(self, model, X_train, y_train, X_test):
		self.X_train = X_train
		self.y_train = y_train
		self.X_test = X_test
		self.model = model

	def cv_scores(self, K=5):
		scores = cross_val_score(self.model, self.X_train, self.y_train, cv=K)
		print("CROSS VALIDATION\nCV Scores: {}\nMean CV Accuracy: {:.3f}".format(scores, scores.mean()))

	@property
	def clf(self):
		_clf = self.model.fit(self.X_train, self.y_train)
		return _clf

	def predict(self, X_test):
		return self.clf.predict(X_test)

	def __str__(self):
		return "Parameters:\n{}".format(self.clf.get_params())

	def run_prediction(self, label_encoder, test_data, file_name):
		self.cv_scores()
		print(str(self.clf))
		y_pred = self.clf.predict(self.X_test)
		y_pred = pd.Series(label_encoder.inverse_transform(y_pred))

		NB_res = pd.DataFrame({'id': test_data['id'], 'cat': y_pred})
		NB_res.to_csv('results/{}-results.csv'.format(file_name), index=False)


def main():
	ip = DataPreprocessing(train_file=cfg['DATA-PATH']['TRAIN'], test_file=cfg['DATA-PATH']['TEST'])
	train_corpus = ip.create_corpus(ip.train_data)
	test_corpus = ip.create_corpus(ip.test_data)
	corpus = train_corpus + test_corpus
	ip.create_target(ip.train_data)

	if cfg['PRE-PROCESS']['COUNT-VECTORIZE']:
		ip.count_vectorize(corpus)
	elif cfg['PRE-PROCESS']['TFIDF-VECTORIZE']:
		ip.tfidf_vectorize(corpus)
	if cfg['PRE-PROCESS']['REDUCE-DIM']:
		ip.reduce_tokens()
	if cfg['PRE-PROCESS']['ONE-HOT']:
		ip.one_hot_encode()

	X_train, y_train = ip.shuffle_train()
	print(str(ip))

	if cfg['CLASSIFIER']['NAIVE-BAYES']['USE']:
		print("\n===== NAIVE BAYES CLASSIFIER =====")
		NB_clf = Classifier(MultinomialNB(alpha=cfg['CLASSIFIER']['NAIVE-BAYES']['ALPHA'], fit_prior=cfg['CLASSIFIER']['NAIVE-BAYES']['FIT-PRIOR']), X_train, y_train, ip.X_test)
		NB_clf.run_prediction(ip.label_encoder, ip.test_data, 'Naive-Bayes')

	if cfg['CLASSIFIER']['SVM']['USE']:
		print("\n===== SVM CLASSIFIER =====")
		SVM_clf = Classifier(SVC(kernel=cfg['CLASSIFIER']['SVM']['KERNEL'], C=cfg['CLASSIFIER']['SVM']['C']), X_train, y_train, ip.X_test)
		SVM_clf.run_prediction(ip.label_encoder, ip.test_data, 'SVM')

if __name__ == '__main__':
	main()





