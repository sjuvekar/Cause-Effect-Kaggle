Cause Effect Pairs Challenge
============================

This repo contains a benchmark and sample code in Python for the [Cause Effect Pairs Challenge](https://www.kaggle.com/c/cause-effect-pairs), a machine learning challenged hosted by [Kaggle](https://www.kaggle.com) and organized by [ChaLearn](http://www.chalearn.org/).

This version of the repo contains the **Basic Python Benchmark**. Future benchmarks may be included here as well and will be marked with git tags.

Executing this benchmark requires Python 2.7 along with the following packages:

 - pandas (tested with version 10.1)
 - sklearn (tested with version 0.13)
 - numpy (tested with version 1.6.2)
 - scipy (tested with version 0.10.)

To run the benchmark,

1. [Download the data](https://www.kaggle.com/c/cause-effect-pairs/data)
2. Modify SETTINGS.json to point to the training and validation data on your system, as well as a place to save the trained model and a place to save the submission
3. Train the model by running `python train_modified.py`
4. Make predictions on the validation set by running `python predict.py`
5. [Make a submission](https://www.kaggle.com/c/cause-effect-pairs/team/select) with the output file

