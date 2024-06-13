from nltk.stem.porter import PorterStemmer 
import numpy as np
import re
import pickle 
from nltk.corpus import stopwords
from utils.string import remove_stopwords_then_stem

class SentimentAnalysisService:

    @classmethod
    def predict(self, texts: list[str]) -> list[dict]:
        path = "./pickles"
        vectorizerFilename = "tfidf_vectorizer.pickle"
        modelFilename = "affectus-analysis.sav"
        
        vectorizer = pickle.load(open(path+"/"+vectorizerFilename ,"rb"))
        texts = list(map(lambda text: remove_stopwords_then_stem(text), texts))
        transformedTexts = vectorizer.transform(texts)

        model = pickle.load(open(path+"/"+modelFilename, "rb"))
        predictions = model.predict(transformedTexts)

        analyzed_sentiments = []
        for i in range(len(predictions)):
            sentiment_code = predictions[i]
            sentiment_message = "Positive" if sentiment_code == 1 else "Negative"
            analyzed_sentiments.append({
                "text": texts[i],
                "code": str(sentiment_code),
                "message": sentiment_message,
            })

        return analyzed_sentiments