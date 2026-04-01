from django.shortcuts import render

# Create your views here.

from django.http import JsonResponse


from .services.interpreter import interpret
from .services.sentiment import analyze_sentiment
from .services.state import update_state

def process_request(request):
    text = request.GET.get("text", "hello")  # simple input

    intent = interpret(text)
    sentiment = analyze_sentiment(text)
    state = update_state(text, sentiment, intent)

    return JsonResponse({
        "text": text,
        "intent": intent,
        "sentiment": sentiment,
        "state": state
    })
