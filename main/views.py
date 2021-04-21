from django.contrib import messages
from django.shortcuts import render

# Create your views here.
from main.classifiers import *


def index(request):
    context = {}
    return render(request, 'main/index.html', context)


def analyse(request, pk):
    model = ""
    if   pk == 1:
        model = 'Logistic Regression'
    elif pk == 2:
        model = 'Naive Bayers'
    elif pk == 3:
        model = 'Support Vector Machine '
    elif pk == 4:
        model = 'K near neighbor'
    elif pk == 5:
        model = 'Decision tree'
    elif pk == 6 or pk == 0:
        model = 'Random Forest'
    global prediction
    if request.POST:
        age = request.POST.get('age')
        sex = request.POST.get('sex')
        cp = request.POST.get('cp')
        restbp = request.POST.get('restbp')
        chol = request.POST.get('chol')
        fbs = request.POST.get('fbs')
        restecg = request.POST.get('restecg')
        thalach = request.POST.get('thalach')
        exang = request.POST.get('exang')
        oldpeak = request.POST.get('oldpeak')
        slope = request.POST.get('slope')
        ca = request.POST.get('ca')
        thal = request.POST.get('thal')
        if pk == 0 or pk == 1:


            prediction = LR_predict(
                [[age, sex, cp, restbp, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])




        elif pk == 2:

            prediction = NB_predict(
                [[age, sex, cp, restbp, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])


        elif pk == 3:


            prediction = SV_predict(
                [[age, sex, cp, restbp, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])


        elif pk == 4:



            prediction = KNN_predict(
                [[age, sex, cp, restbp, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])



        elif pk == 5:

            prediction = DT_predict(
                [[age, sex, cp, restbp, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])


        elif pk == 6:

            prediction = RF_predict(
                [[age, sex, cp, restbp,chol, fbs, restecg,thalach, exang, oldpeak,slope, ca, thal]])

        output = ""
        if prediction == 0:
            output = 'You have least possibility to have a heart disease'
        elif prediction == 1:
            output = 'Sorry to say that you have a heart disease , we recommend you to consult a doctor'
        messages.success(request, output)

    context = {'model': model}
    return render(request, 'main/analyse.html', context)
