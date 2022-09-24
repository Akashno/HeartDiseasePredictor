from django.contrib import messages
from django.http import JsonResponse
from django.shortcuts import render

# Create your views here.
from main.classifiers import *
from main.forms import DatasetForm
from main.models import Train, Dataset
from main.train import train_main


def index(request):
    context = {}
    return render(request, 'main/index.html', context)

def home(request):
    context = {}
    return render(request, 'main/home.html', context)

def learnMore(request):
    faqs = [
         { 'id':'0','title':"What is heart disease predictor",'content':"This project analyzes the heart disease patient dataset with proper data processing. Then, different models are trained and predictions are made based on that models. Classifiers like Logistic regression, Naive Bayes, Decision trees, Random forest, and support vector machines are used as model algorithms for predicting the results. These classifiers were imported from the sci-kit learn library which is a famous library for machine learning problems in python. The output target is predicted from the different models used by giving the 13 predictor variables such as Age, Sex, Angina, Resting Blood Pressure, Serum cholesterol, Fasting Blood Sugar, Resting ECG, Max heart rate achieved, Exercise-induced angina, Peak exercise St segment."},
         { 'id':'1','title':"Classifiers used to predict the disease",'content':"There is a total of 6 Algorithms used Here for Model Training.  Logistic regression Algorithm Gaussian Naive Bayes Algorithm Decision tree Algorithm Random Forest Algorithm Support vector machines Algorithm KNN Algorithm These ALgorithms are imported from the sci-kit learn library which is a famous library for machine learning problems in python."}
    ]
    context = { 'faqs':faqs }
    return render(request, 'main/learnMore.html', context)

def analyse(request, pk):
    data = {
    'lr':'Logistic Regression','nb':'Naive Bayers','sv':'Support Vector Machine ',
    'knn':'K near neighbor' ,'dt':'Decision tree','rf':'Random Forest'
    }
    model = ""
    if pk == 0:
        f = open("main/ML_models/accurate.txt", "r")
        accurate = f.read()
        model = data[accurate]
    elif pk == 1:
        model = 'Logistic Regression'
    elif pk == 2:
        model = 'Naive Bayers'
    elif pk == 3:
        model = 'Support Vector Machine '
    elif pk == 4:
        model = 'K near neighbor'
    elif pk == 5:
        model = 'Decision tree'
    elif pk == 6 :
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

        if model == 'Logistic Regression':
            prediction = LR_predict(
                [[age, sex, cp, restbp, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        elif model == 'Naive Bayers':
            prediction = NB_predict(
                [[age, sex, cp, restbp, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        elif model  == 'Support Vector Machine ':
            prediction = SV_predict(
                [[age, sex, cp, restbp, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        elif model  == 'K near neighbor':
            prediction = KNN_predict(
                [[age, sex, cp, restbp, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        elif model == 'Decision tree':
            prediction = DT_predict(
                [[age, sex, cp, restbp, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        elif model == 'Random Forest':
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




def train(request):
    form = DatasetForm()
    if request.POST:
        form = DatasetForm(request.POST , request.FILES)
        if form.is_valid():
            form.save()
            object = Train.objects.last()
            if object.training:
                return JsonResponse('training', safe=False)
            else:
                object.training = True
                object.save()
                try:
                   train_main()
                except:
                    object.training = False
                    object.save()
                    data = Dataset.objects.last()
                    data.file.delete()
                    data.delete()
                    response = JsonResponse("error")
                    response.status_code = 403  # To announce that the user isn't allowed to publish
                    return response
                object.training = False
                object.save()
                return JsonResponse('trained', safe=False)
        else:
            return JsonResponse('invalidFile', safe=False)
    context = {'form':form}
    return render(request, 'main/train.html', context)