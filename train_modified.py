import data_io
import pickle
import feature_extractor as fe
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.grid_search import GridSearchCV
import pandas as pd
import csv
from time import time
import numpy

def get_pipeline():
    features = fe.feature_extractor()
    classifier = GradientBoostingClassifier(n_estimators=1536,
                                          random_state = 1,
                                          subsample = .8,
                                          min_samples_split=10,
                                          max_depth = 6,
                                          verbose=3)
    steps = [("extract_features", features),
             ("classify", classifier)]
    myP = Pipeline(steps)
#    params = {"classify__n_estimators": [768, 1024, 1536], "classify__min_samples_split": [1, 5, 10], "classify__min_samples_leaf": [1, 5, 10]}
#    grid_search = GridSearchCV(myP, params, n_jobs=8)
#    return grid_search
#   return myP
    return (features, classifier)


def get_types(data):
    data['Bin-Bin'] = (data['A type']=='Binary')&(data['B type']=='Binary')
    data['Num-Num'] = (data['A type']=='Numerical')&(data['B type']=='Numerical')
    data['Cat-Cat'] = (data['A type']=='Categorical')&(data['B type']=='Categorical')

    data[['A type','B type']] = data[['A type','B type']].replace('Binary',1)
    data[['A type','B type']] = data[['A type','B type']].replace('Categorical',1)
    data[['A type','B type']] = data[['A type','B type']].replace('Numerical',0)
    return data

def combine_types(data, data_info):
    data = pd.concat([data,data_info],axis = 1)
    types = []
    for a,b in zip(data['A type'], data['B type']):
        types.append(a + b)
    data['types'] = types
    #data['types'] = [x + y for x in data['A type'] for y in data['B type']]
    return data

"""
Return one classifier for one catagory
"""
def classify_catagory(train, test):
    print("Train-test split")
    trainX, testX, trainY, testY = train_test_split(train, test, random_state = 1)
    print "TrainX size = ", str(trainX.shape)
    print "TestX size = ", str(testX.shape)

    classifier = GradientBoostingClassifier(n_estimators=1536,
                                          random_state = 1,
                                          subsample = .8,
                                          min_samples_split=10,
                                          max_depth = 6,
                                          verbose=3)
    classifier.fit(trainX, trainY)
    print "Score = ", classifier.score(testX, testY)

    feature_importrance = classifier.feature_importances_
    logger = open(data_io.get_paths()["feature_importance_path"], "a")
    for fi in feature_importrance:
      logger.write(str(fi))
      logger.write("\n")
    logger.write("###########################################\n")
    logger.close()

    return classifier
 

""" 
categories are: Num+Num, Num+~Num, ~NUm+Num, ~Num+~Num
"""
def create_classifiers(train, test, train_info):
    num_num = (train_info["A type"] == "Numerical") & (train_info["B type"] == "Numerical")
    num_num_classifier = classify_catagory(train[num_num], test[num_num])

    num_not_num = (train_info["A type"] == "Numerical") & (train_info["B type"] != "Numerical")
    num_not_num_classifier = classify_catagory(train[num_not_num], test[num_not_num])

    not_num_num = (train_info["A type"] != "Numerical") & (train_info["B type"] == "Numerical")
    not_num_num_classifier = classify_catagory(train[not_num_num], test[not_num_num])

    not_num_not_num = (train_info["A type"] != "Numerical") & (train_info["B type"] != "Numerical")
    not_num_not_num_classifier = classify_catagory(train[not_num_not_num], test[not_num_not_num])

    return (num_num_classifier, num_not_num_classifier, not_num_num_classifier, not_num_not_num_classifier)


def main():
    t1 = time()
    print("Reading in the training data")
    train = data_io.read_train_pairs()
    train_info = data_io.read_train_info()
    train = combine_types(train, train_info)

    #make function later
    train = get_types(train)
    target = data_io.read_train_target()

    print "Reading SUP data..."
    for i in range(1,4):
      print "SUP", str(i)
      sup = data_io.read_sup_pairs(i)
      sup_info = data_io.read_sup_info(i)
      sup = combine_types(sup, sup_info)
      sup = get_types(sup)
      sup_target = data_io.read_sup_target(i)
      train_info = train_info.append(sup_info)
      train = train.append(sup)
      target = target.append(sup_target)

    # Old train
    print "Reading old train data..."
    old_train = data_io.read_old_train_pairs()
    old_train_info = data_io.read_old_train_info()
    old_train = combine_types(old_train, old_train_info)
    old_train = get_types(old_train)
    old_target = data_io.read_old_train_target()

    train = train.append(old_train)
    target = target.append(old_target)
    # End old train

    print "Train size = ", str(train.shape)
    print("Extracting features and training model")
    feature_trans = fe.feature_extractor()
    orig_train = feature_trans.fit_transform(train)
    orig_train = numpy.nan_to_num(orig_train) 

    classifier = classify_catagory(orig_train, target.Target)
    #(both_classifier, A_classifier, B_classifier, none_classifier) = create_classifiers(orig_train, target.Target, train_info)

    print("Saving features")
    data_io.save_features(orig_train)

    print("Saving the classifier")
    #data_io.save_model( (both_classifier, A_classifier, B_classifier, none_classifier) )
    data_io.save_model(classifier) 
 
    #features = [x[0] for x in classifier.steps[0][1].features ]

    #csv_fea = csv.writer(open('features.csv','wb'))
    #imp = sorted(zip(features, classifier.steps[1][1].feature_importances_), key=lambda tup: tup[1], reverse=True)
    #for fea in imp:
    #    print fea[0], fea[1]
    #    csv_fea.writerow([fea[0],fea[1]])


    t2 = time()
    t_diff = t2 - t1
    print "Time Taken (min):", round(t_diff/60,1)

if __name__ == "__main__":
  main()
