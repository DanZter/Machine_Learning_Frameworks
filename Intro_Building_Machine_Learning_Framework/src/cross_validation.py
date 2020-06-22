import pandas as pd
from sklearn import model_selection

"""
- binary_classification
- multi-class classification
- multi-label classification
- single column regression
- multi column regression
- holdout
"""


class CrossValidation:
    def __init__(
            self,
            df,
            target_cols,
            problem_type="binary_classification",
            num_folds=5,
            shuffle=True,
            random_state=42
        ):
        self.dataframe = df
        self.target_cols = target_cols
        self.num_targets = len(target_cols)
        self.problem_type = problem_type
        self.num_folds = num_folds
        self.shuffle = shuffle
        self.random_state = random_state

        if self.shuffle is True:
            self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)

        self.dataframe["kfold"] = -1

    def split(self):
        if self.problem_type == "binary_classification":
            target = self.target_cols[0]
            unique_values = self.dataframe[target].nunique()
            if unique_values == 1:
                raise Exception("Only one Unique Target value found!")
            elif unique_values > 1:

                kf = model_selection.StratifiedKFold(n_splits=self.num_folds,
                                                     shuffle=False)
                for fold, (train_idx, val_idx) in enumerate(
                        kf.split(X=self.dataframe, y=self.dataframe[target].values)):
                    self.dataframe.loc[val_idx, 'kfold'] = fold

        return self.dataframe


if __name__ == "__main__":
    df = pd.read_csv("./input/train.csv")
    cv = CrossValidation(df, target_cols=["target"])
    df_split = cv.split()
    print(df_split.head())
    print(df_split[df_split.kfold == 0].target.value_counts())
    print(df_split[df_split.kfold == 1].target.value_counts())
    print(df_split.kfold.value_counts())