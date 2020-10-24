from django.shortcuts import render
from django.http import HttpResponse
from news_site.proximity import main, proximity

# Create your views here.
# def index(request):
#     return HttpResponse("Hello, world. You're at the polls index.")

def home(request):
    return render(request, 'index.html')

def result(request):
    return render(request, 'result.html')

def result(request):
    if request.method == 'POST':
        query = request.POST['querysearch']
        hasil = main.hello(query)
        proximitys = proximity.results(query)

        content={
	    'proximitys': proximitys,
            'hasil':hasil,
            'query':query
        }
        return render(request, 'result.html',content)