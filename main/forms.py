from django import forms
from django.core.exceptions import ValidationError
from pandas.io.sas.sas_constants import magic

from main.models import Dataset


class DatasetForm(forms.ModelForm):
    class Meta:
        model = Dataset
        fields = ['name', 'file']