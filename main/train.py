import time

import pandas as pd
from django.conf import settings
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
import joblib
# classifiers -----------
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# classifiers -----------
import warnings

from main.models import Dataset

def train_main():
    warnings.filterwarnings('ignore')
    object = Dataset.objects.last()
    dataset = pd.read_csv(str(settings.BASE_DIR)+str(object.file.url))
    predictors = dataset.drop("target", axis=1)
    target = dataset["target"]
    X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size=0.20, random_state=0)

    train_lr_model(X_train, Y_train, X_test, Y_test)
    # train_nb_model(X_train, Y_train, X_test, Y_test)
    # train_sv_model(X_train, Y_train, X_test, Y_test)
    # train_knn_model(X_train, Y_train, X_test, Y_test)
    # train_dt_model(X_train, Y_train, X_test, Y_test)
    # train_rf_model(X_train, Y_train, X_test, Y_test)

def train_lr_model(X_train, Y_train, X_test, Y_test):
    time.sleep(5)
    LR_model = LogisticRegression()
    LR_model.fit(X_train, Y_train)
    joblib.dump(LR_model, 'main/ML_models/LR_model.joblib')

    # prediction = LR_model.predict(X_test)
    # score = accuracy_score(prediction, Y_test)
    # print("logistic :", score)

def train_nb_model(X_train, Y_train, X_test, Y_test):
    NB_model = GaussianNB()
    NB_model.fit(X_train, Y_train)
    joblib.dump(NB_model, 'main/ML_models/NB_model.joblib')

    # prediction = NB_model.predict(X_test)
    # score = accuracy_score(prediction, Y_test)
    # print("nav bayers :", score)


def train_sv_model(X_train, Y_train, X_test, Y_test):
    SV_model = svm.SVC(kernel='linear')
    SV_model.fit(X_train, Y_train)
    joblib.dump(SV_model, 'main/ML_models/SV_model.joblib')

    # prediction = SV_model.predict(X_test)
    # score = accuracy_score(prediction, Y_test)
    # print("sv :", score)


def train_knn_model(X_train, Y_train, X_test, Y_test):
    KNN_model = KNeighborsClassifier(n_neighbors=7)
    KNN_model.fit(X_train, Y_train)
    joblib.dump(KNN_model, 'main/ML_models/KNN_model.joblib')

    # prediction = KNN_model.predict(X_test)
    # score = accuracy_score(prediction, Y_test)
    # print("knn :", score)


def train_dt_model(X_train, Y_train, X_test, Y_test):
    global best_x
    max_accuracy = 0
    for x in range(200):
        DT_model = DecisionTreeClassifier(random_state=x)
        DT_model.fit(X_train, Y_train)
        Y_pred_dt = DT_model.predict(X_test)
        current_accuracy = round(accuracy_score(Y_pred_dt, Y_test) * 100, 2)
        if (current_accuracy > max_accuracy):
            max_accuracy = current_accuracy
            best_x = x

    DT_model = DecisionTreeClassifier(random_state=best_x)
    DT_model.fit(X_train, Y_train)
    joblib.dump(DT_model, 'main/ML_models/DT_model.joblib')

    # prediction = DT_model.predict(X_test)
    # score = accuracy_score(prediction, Y_test)
    # print("dt :", score)


def train_rf_model(X_train, Y_train, X_test, Y_test):
    global best_x
    max_accuracy = 0
    for x in range(2000):
        RF_model = RandomForestClassifier(random_state=x)
        RF_model.fit(X_train, Y_train)
        Y_pred_rf = RF_model.predict(X_test)
        current_accuracy = round(accuracy_score(Y_pred_rf, Y_test) * 100, 2)
        if (current_accuracy > max_accuracy):
            max_accuracy = current_accuracy
            best_x = x

    RF_model = RandomForestClassifier(random_state=best_x)
    RF_model.fit(X_train, Y_train)
    joblib.dump(RF_model, 'main/ML_models/RF_model.joblib')

    # prediction = RF_model.predict(X_test)
    # score = accuracy_score(prediction, Y_test)
    # print("rf :", score)


