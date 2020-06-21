import os
import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import joblib

# from . import dispatcher
# TEST_DATA = os.environ.get("TEST_DATA")
# MODEL = os.environ.get("MODEL")

TRAINING_DATA = "input/train_folds.csv"
TEST_DATA = "input/test.csv"
MODEL = "randomforest"
MODELS = {
    "randomforest": ensemble.RandomForestClassifier(n_estimators=200, n_jobs=4, verbose=2),
    "extratrees": ensemble.ExtraTreesClassifier(n_estimators=200, n_jobs=4, verbose=2)
}

def predict():
    test_df = pd.read_csv(TEST_DATA)
    test_idx = test_df["id"].values
    predictions = None

    for FOLD in range(5):
        print(FOLD)
        test_df = pd.read_csv(TEST_DATA)
        encoders = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}_label_encoder.pkl"))
        cols = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}_columns.pkl"))
        for c in cols:
            print(c)
            lbl = encoders[c]
            test_df.loc[:, c] = lbl.transform(test_df[c].values.tolist())

        clf = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}.pkl"))

        test_df = test_df[cols]
        preds = clf.predict_proba(test_df)[:, 1]

        if FOLD == 0:
            predictions = preds
        else:
            predictions += preds

    predictions /= 5

    sub = pd.DataFrame(np.column_stack((test_idx, predictions)), columns = ["id", "target"])
    return sub

if __name__ == "__main__":
    submission = predict()
    submission.id = submission.id.astype(int)
    submission.to_csv(f"models/{MODEL}.csv", index = False)

