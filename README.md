# Quiz-Bowl-Champion

Building an AI that can categorize Quiz bowl questions.

## How to Run

This project assumes that you have Python 3.6 installed. Additional packages required to run the project without errors can be installed by running `pip intall -r requirements.txt`.


To generate the inference on the test data, go to the project root directory and run:
```
python3 src/pipeline.py
```

The path to the files, steps for data pre-processing (feature engineering), and the type of classifier can be specified using the `config.yaml` file. Currently, the implementation supports prediction only using the [Naive Bayes Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html).

## Data
The data for this challenge consists of full Quiz Bowl question text and the category that each question belongs to.  

The training data file, `coggle_train.csv` includes the question id, the question text, and the category.  The testing data file, `coggle_test.csv` only contains question id and text.  

### File descriptions
* `coggle_train.csv` - the training set - header: [`id, text, cat`]
* `coggle_test.csv` - the test set - header: [`id, text`]

### Data fields
* `id` - integer - unique question identifier
* `text` - string - text from Quiz Bowl question
* `cat` - string - the category of the question

