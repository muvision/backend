from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from PIL import Image
from .functions import classification


def index(request):
    return HttpResponse(f"Hello, world. Youre at the muvision index.")


def classify_single(request):
    if request.method == "POST":
        print(request.FILES['image'])
        image_file = Image.open(request.FILES['image'])
        res = classification.classify(image_file)
        return HttpResponse(f"This character is probably {res}")
    return HttpResponse(f"Hello, world. Youre at the classification page. Please make a post request with the image in the body")

def classify_image(request):
    if request.method == "POST":
        print(request.FILES['image'])
        image_file = Image.open(request.FILES['image'])
        res = classification.classify(image_file)
        return HttpResponse(f"This character is probably {res}")
    return HttpResponse(f"Hello, world. Youre at the classification page. Please make a post request with the image in the body")