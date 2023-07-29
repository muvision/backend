from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from PIL import Image
from functions.image_reader import image_reader
from .functions import classification


def index(request):
    return HttpResponse(f"Hello, world. Youre at the muvision index.")


def classify_image(request):
    if request.method == "POST":
        print(request.FILES['image'])
        image_file = Image.open(request.FILES['image'])
        document = image_reader(image_file)
        return HttpResponse(f"This character is probably yo mama")
    return HttpResponse(f"Hello, world. Youre at the classification page. Please make a post request with the image in the body")