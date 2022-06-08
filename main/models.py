from django.db import models

# Create your models here.
class Dataset(models.Model):
    name = models.CharField(max_length=100,null=True)
    file = models.FileField(upload_to='train_dataset', null=True)
    date_created = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return str(self.name)

class Train(models.Model):
    training = models.BooleanField(default=False,null=True)
    def __str__(self):
        return str(self.training)
 