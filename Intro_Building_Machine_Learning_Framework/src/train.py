import os
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import joblib

# from . import dispatcher
# TRAINING_DATA = os.environ.get("TRAINING_DATA")
# TEST_DATA = os.environ.get("TEST_DATA")
# FOLD = int(os.environ.get("FOLD"))
# MODEL = os.environ.get("MODEL")

TRAINING_DATA = "input/train_folds.csv"
TEST_DATA = "input/test.csv"
MODEL = "randomforest"
MODELS = {
    "randomforest": ensemble.RandomForestClassifier(n_estimators=200, n_jobs=4, verbose=2),
    "extratrees": ensemble.ExtraTreesClassifier(n_estimators=200, n_jobs=4, verbose=2)
}

FOLD_MAPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3]
}
if __name__ == "__main__":
    for FOLD in range(5):
        df = pd.read_csv(TRAINING_DATA)
        test_df = pd.read_csv(TEST_DATA)

        train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))]
        valid_df = df[df.kfold == FOLD]

        ytrain = train_df.target.values
        yvalid = valid_df.target.values

        train_df = train_df.drop(['id', 'target', 'kfold'],     axis=1)  # dropping unnecessary features
        valid_df = valid_df.drop(['id', 'target', 'kfold'], axis=1)

        valid_df = valid_df[train_df.columns]  # order of columns are corrected w.r.t training data

        label_encoders = {}  # label encoding
        for c in train_df.columns:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(train_df[c].values.tolist() + valid_df[c].values.tolist() + test_df[c].values.tolist())
            train_df.loc[:, c] = lbl.transform(train_df[c].values.tolist())
            valid_df.loc[:, c] = lbl.transform(valid_df[c].values.tolist())
            label_encoders[c] = lbl

# DATA IS READY TO TRAIN

        clf = MODELS[MODEL]
        # clf = dispatcher.MODELS[MODEL]
        clf.fit(train_df, ytrain)
        preds = clf.predict_proba(valid_df)[:, 1]
        print(metrics.roc_auc_score(yvalid, preds))

        joblib.dump(label_encoders, f"models/{MODEL}_{FOLD}_label_encoder.pkl")
        joblib.dump(clf, f"models/{MODEL}_{FOLD}.pkl")
        joblib.dump(train_df.columns, f"models/{MODEL}_{FOLD}_columns.pkl")