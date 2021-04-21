import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')


def LR_predict(X):
    LR_model = joblib.load('main/ML_models/LR_model.joblib')
    LR_predict: object = LR_model.predict(X)
    return LR_predict


def NB_predict(X):
    NB_model = joblib.load('main/ML_models/NB_model.joblib')
    NB_predict = NB_model.predict(X)
    return NB_predict


def SV_predict(X):
    SV_model = joblib.load('main/ML_models/SV_model.joblib')
    SV_predict = SV_model.predict(X)
    return SV_predict


def KNN_predict(X):
    KNN_model = joblib.load('main/ML_models/KNN_model.joblib')
    KNN_predict = KNN_model.predict(X)
    return KNN_predict


def DT_predict(X):
    DT_model = joblib.load('main/ML_models/DT_model.joblib')
    DT_predict = DT_model.predict(X)
    return DT_predict


def RF_predict(X):
    RF_model = joblib.load('main/ML_models/RF_model.joblib')
    RF_predict = RF_model.predict(X)
    return RF_predict
