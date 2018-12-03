# Quiz-Bowl-Champion

Building an AI that can categorize Quiz bowl questions.

## How to Run

This project assumes that you have Python 3.6 installed. Additional packages required to run the project without errors can be installed by running `pip intall -r requirements.txt`.


To generate the inference on the test data, go to the project root directory and run:
```
python3 src/pipeline.py
```

The result obtained after running inference in the test data is stored in the `results` folder. For example, the `Naive-Bayes-results.csv` file contains the results obtained using the Naive Bayes classifier with the header: [`id, cat`].  

The path to the files, steps for data pre-processing (feature engineering), and the type of classifier can be specified using the `config.yaml` file. Currently, the implementation supports prediction only using the [Naive Bayes Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html).

## Data
The data for this challenge consists of full Quiz Bowl question text and the category that each question belongs to.  

The training data file, `coggle_train.csv` includes the question id, the question text, and the category.  The testing data file, `coggle_test.csv` only contains question id and text.  

### File descriptions
- `coggle_train.csv` - the training set - header: [`id, text, cat`]
- `coggle_test.csv` - the test set - header: [`id, text`]

### Data fields
- `id` - integer - unique question identifier
- `text` - string - text from Quiz Bowl question
- `cat` - string - the category of the question

## Feature Engineering
Since questions in the dataset are a sequence of words, they need to be converted into numerical feature vectors. I used the `bag of words` model in which each sentence is divided into words and the number of occurrences of each word in the sentence is counted. Each word is then assigned a number which the number of times the word appears in the sentence. Each unique word obtained from the entire corpus corresponds to a feature. These feature vectors (Document-term matrix) can be created using the [`CountVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) module in `sklearn`.  

Additionally, the `CountVectorizer` module removes punctuations and stop-words from the corpus as they do not provide much information and are not very discriminative. Removing them also reduces the number of features.  

The weightage of common words can further be reduced using TF-IDF i.e Term Frequency times inverse document frequency instead of just the count. Details about TF-IDF can be found [here](http://www.tfidf.com/). `sklearn` provides an easy to use module [`TfidfVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) to achieve this.  

## Classifiers
So far, I have trained the following classifiers on the Quiz bowl dataset:
- [x] **Naive Bayes:** using the [Multinomial Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html) classifier in `sklearn`
- [ ] **Support Vector Machine**

## Results
- [x] **Naive Bayes:** 5-fold mean cross-validation accuracy = 79.02%

## Work in Progress
- [ ] Using stemming to reduce the number of features
- [ ] Using a deep learning-based model such as [word2vec](https://www.tensorflow.org/tutorials/representation/word2vec) to create features (word embeddings)



