import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

from sklearn import preprocessing
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model, load_model


class EntityEmbeddings:
    def __init__(self, train, test, features):
        self.train = train
        self.test = test
        self.features = features

        self.test.loc[:, 'target'] = -1
        data = pd.concat([self.train, self.test]).reset_index(drop=True)
        print(data.shape, self.train.shape, self.test.shape)

        for feat in self.features:
            lbl_enc = preprocessing.LabelEncoder()
            data.loc[:, feat] = lbl_enc.fit_transform(data[feat].astype(str).fillna("-1").values)
        print(data.head())

        self.train = data[data.target != -1].reset_index(drop=True)
        self.test = data[data.target == -1].reset_index(drop=True)
        print(data.shape, self.train.shape, self.test.shape)

    def get_model(self):
        inputs = []
        outputs = []
        for c in self.features:
            num_unique_vals = int(self.train[c].nunique())
            embed_dim = int(min(np.ceil(num_unique_vals / 2), 50))
            inp = layers.Input(shape=(1,))
            out = layers.Embedding(num_unique_vals + 1, embed_dim, name=c)(inp)
            # apply dropouts here
            out = layers.Reshape(target_shape=(embed_dim,))(out)
            inputs.append(inp)
            outputs.append(out)
        x = layers.Concatenate()(outputs)
        x = layers.Dense(300, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        y = layers.Dense(1, activation='sigmoid')(x)
        model = Model(inputs=inputs, outputs=y)
        return model



if __name__ == "__main__":

    df_train = pd.read_csv("./input/train_classification_2_categorical_feature_encoding.csv")  # .head(500)
    df_test = pd.read_csv("./input/test_classification_2_categorical_feature_encoding.csv")  # .head(500)
    sample = pd.read_csv("./input/sample_submission_2_categorical_feature_encoding.csv")

    col_features = [f for f in df_train.columns if f not in ["id", "target"]]
    print(col_features)

    mdl = EntityEmbeddings(df_train, df_test, col_features)
    # mdl.get_model().summary()

    model = mdl.get_model()
    print(model)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    print([df_train.head(100).loc[:, f].values for f in col_features])
    # model.fit([df_train.loc[:, f] for f in col_features], df_train.target.values)
