#!/usr/bin/env python

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.externals import joblib  #模型持久化

def get_sepsis_score(data, model):
    #model = load_sepsis_model()
    data_mean = ([
        83.8996, 97.0520, 36.8055, 126.2240, 86.2907,
        66.2070, 18.7280, 33.7373, -3.1923, 22.5352,
        0.4597, 7.3889, 39.5049, 96.8883, 103.4265,
        22.4952, 87.5214, 7.7210, 106.1982, 1.5961,
        0.6943, 131.5327, 2.0262, 2.0509, 3.5130,
        4.0541, 1.3423, 5.2734, 32.1134, 10.5383,
        38.9974, 10.5585, 286.5404, 198.6777,23,24,25,26,29,10])
    values = np.vstack((data_mean,data))
    values = pd.DataFrame(values)

    # 特征选取 去除缺省率大于99%
    values.drop(values.columns[[7, 13, 14, 16, 20, 26, 27, 32, 36, 37]], axis=1, inplace=True)

    values = values.fillna(method='pad')  # 缺省值填充为前一个值

    maxmin = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(values)
    values_norm = maxmin.transform(values)

    x_test = values_norm
    prediction_probas = model.predict_proba(x_test)
    prediction_proba = prediction_probas[-1]
    labels = model.predict(x_test)
    label = labels[-1]
    if label == 1:
        score = max(prediction_proba)
    else:
        score = min(prediction_proba)

    return score, label

def load_sepsis_model():

    model = joblib.load('clf_SGDr25000.pkl')

    return model
