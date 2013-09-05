import csv
import json
import numpy as np
import os
import pandas as pd
import pickle

def get_paths():
    paths = json.loads(open("SETTINGS.json").read())
    for key in paths:
        paths[key] = os.path.expandvars(paths[key])
    return paths

def parse_dataframe(df):
    parse_cell = lambda cell: np.fromstring(cell, dtype=np.float, sep=" ")
    df = df.applymap(parse_cell)
    return df

def read_train_pairs():
    train_path = get_paths()["train_pairs_path"]
    return parse_dataframe(pd.read_csv(train_path, index_col="SampleID"))

def read_sup_pairs(i):
    sup_prefix = get_paths()["sup_prefix"]
    return parse_dataframe(pd.read_csv(sup_prefix + str(i) + "data_text/CEdata_train_pairs.csv", index_col="SampleID"))

def read_old_train_pairs():
    train_path = get_paths()["old_train_pairs_path"]
    return parse_dataframe(pd.read_csv(train_path, index_col="SampleID"))


def read_train_target():
    path = get_paths()["train_target_path"]
    df = pd.read_csv(path, index_col="SampleID")
    df = df.rename(columns = dict(zip(df.columns, ["Target", "Details"])))
    return df

def read_sup_target(i):
      path = get_paths()["sup_prefix"]
      df = pd.read_csv(path + str(i) + "data_text/CEdata_train_target.csv", index_col="SampleID")
      df = df.rename(columns = dict(zip(df.columns, ["Target", "Details"])))
      return df

def read_old_train_target():
    path = get_paths()["old_train_target_path"]
    df = pd.read_csv(path, index_col="SampleID")
    df = df.rename(columns = dict(zip(df.columns, ["Target", "Details"])))
    return df


def read_train_info():
    path = get_paths()["train_info_path"]
    return pd.read_csv(path, index_col="SampleID")

def read_sup_info(i):
      path = get_paths()["sup_prefix"]
      return pd.read_csv(path + str(i) + "data_text/CEdata_train_publicinfo.csv", index_col="SampleID")

def read_old_train_info():
    path = get_paths()["old_train_info_path"]
    return pd.read_csv(path, index_col="SampleID")


def read_valid_pairs():
    valid_path = get_paths()["valid_pairs_path"]
    return parse_dataframe(pd.read_csv(valid_path, index_col="SampleID"))

def read_valid_info():
    path = get_paths()["valid_info_path"]
    return pd.read_csv(path, index_col="SampleID")

def read_test_pairs():
    test_path = get_paths()["test_pairs_path"]
    return parse_dataframe(pd.read_csv(test_path, index_col="SampleID"))

def read_test_info():
    path = get_paths()["test_info_path"]
    return pd.read_csv(path, index_col="SampleID")

def read_solution():
    solution_path = get_paths()["solution_path"]
    return pd.read_csv(solution_path, index_col="SampleID")

def save_model(model):
    out_path = get_paths()["model_path"]
    pickle.dump(model, open(out_path, "w"))

def save_features(features):
    out_path = get_paths()["feature_path"]
    pickle.dump(features, open(out_path, "w"))

def save_valid_features(features):
    out_path = get_paths()["valid_feature_path"]
    pickle.dump(features, open(out_path, "w"))

def save_test_features(features):
    out_path = get_paths()["test_feature_path"]
    pickle.dump(features, open(out_path, "w"))

def load_model():
    in_path = get_paths()["model_path"]
    return pickle.load(open(in_path))

def read_submission():
    submission_path = get_paths()["submission_path"]
    return pd.read_csv(submission_path, index_col="SampleID")

def write_submission(predictions):
    submission_path = get_paths()["submission_path"]
    writer = csv.writer(open(submission_path, "w"), lineterminator="\n")
    valid = read_valid_pairs()
    rows = [x for x in zip(valid.index, predictions)]
    writer.writerow(("SampleID", "Target"))
    writer.writerows(rows)

def write_test_submission(predictions):
    test_submission_path = get_paths()["test_submission_path"]
    writer = csv.writer(open(test_submission_path, "w"), lineterminator="\n")
    test = read_test_pairs()
    rows = [x for x in zip(test.index, predictions)]
    writer.writerow(("SampleID", "Target"))
    writer.writerows(rows)
