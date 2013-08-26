import data_io
import feature_extractor as fe
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

def get_pipeline():
    features = fe.feature_extractor()
    steps = [("extract_features", features),
             ("scale", StandardScaler()),
             ("classify", RandomForestRegressor(n_estimators=1024, 
                                                verbose=2,
                                                n_jobs=1,
                                                min_samples_split=10,
                                                min_samples_leaf=10,
                                                random_state=1))]
    p = Pipeline(steps)
    params = dict(classify__n_estimators=[768, 1024, 1536], classify__min_samples_split=[1, 5, 10], classify__min_samples_leaf= [1, 5, 10])
    grid_search = GridSearchCV(p, params, n_jobs=8)
    return grid_search

def main():
    print("Reading in the training data")
    train = data_io.read_train_pairs()
    target = data_io.read_train_target()

    print("Extracting features and training model")
    classifier = get_pipeline()
    classifier.fit(train, target.Target)

    print("Saving the classifier")
    data_io.save_model(classifier)
    
if __name__=="__main__":
    main()
