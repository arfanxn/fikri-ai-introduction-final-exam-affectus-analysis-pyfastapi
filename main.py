from fastapi import FastAPI, Response, Form, status
from typing_extensions import Annotated
import json
from app.services.sentiment_analysis_service import SentimentAnalysisService

app = FastAPI()

@app.get("/api/sentiment-analyses")
async def get_sentiment_analysis():
    content = json.dumps({"message": "This url is on an under construction"})
    return Response(status_code=status.HTTP_200_OK, content=content, media_type="application/json")

@app.post("/api/sentiment-analyses")
async def store_sentiment_analysis(text: Annotated[str, Form()]):
    service = SentimentAnalysisService()
    analyzed_sentiments = service.predict([text])
    content = json.dumps({
        "message": "Successfully analyzed sentiments.",
        "analyzed_sentiment" : analyzed_sentiments[0]
    })
    return Response(status_code=status.HTTP_200_OK, content=content, media_type="application/json")