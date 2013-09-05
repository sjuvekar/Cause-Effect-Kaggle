import data_io
import numpy as np
import pickle
import feature_extractor as fe 

def historic():
    print("Calculating correlations")
    calculate_pearsonr = lambda row: abs(pearsonr(row["A"], row["B"])[0])
    correlations = valid.apply(calculate_pearsonr, axis=1)
    correlations = np.array(correlations)

    print("Calculating causal relations")
    calculate_causal = lambda row: causal_relation(row["A"], row["B"])
    causal_relations = valid.apply(calculate_causal, axis=1)
    causal_relations = np.array(causal_relations)

    scores = correlations * causal_relations

def main():
    print("Reading the test pairs") 
    test = data_io.read_test_pairs()
    features = fe.feature_extractor()
    print("Transforming features")
    trans_test = features.fit_transform(test)
    trans_test = np.nan_to_num(trans_test)

    print("Saving Valid Features")
    data_io.save_test_features(trans_test)

    print("Loading the classifier")
    #(both_classifier, A_classifier, B_classifier, none_classifier) = data_io.load_model()
    classifier = data_io.load_model()

    print("Making predictions")
    test_info = data_io.read_test_info() 
    predictions = list()
    curr_pred = None
    """
    for i in range(len(trans_valid)):
      
      if valid_info["A type"][i] == "Numerical" and valid_info["B type"][i] == "Numerical":
        curr_pred = both_classifier.predict_proba(trans_valid[i, :])
      
      elif valid_info["A type"][i] == "Numerical" and valid_info["B type"][i] != "Numerical":
        curr_pred = A_classifier.predict_proba(trans_valid[i, :])
      
      elif valid_info["A type"][i] != "Numerical" and valid_info["B type"][i] == "Numerical":
        curr_pred = B_classifier.predict_proba(trans_valid[i, :])
     
      else:
        curr_pred = none_classifier.predict_proba(trans_valid[i, :])

      predictions.append(curr_pred[0][2] - curr_pred[0][0])
    """

    orig_predictions = classifier.predict_proba(trans_test)
    predictions = orig_predictions[:, 2] - orig_predictions[:, 0]
    predictions = predictions.flatten()

    print("Writing predictions to file")
    data_io.write_test_submission(predictions)

if __name__=="__main__":
    main()
