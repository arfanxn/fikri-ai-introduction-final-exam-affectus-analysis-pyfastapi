import pickle

class SentimentAnalysisService:

    @classmethod
    def predict(self, texts: list[str]) -> list[dict]:
        path = "./resources/kaggle"
        vectorizer = pickle.load(open(path+"/affectus-analysis-vectorizer.pickle","rb"))
        model = pickle.load(open(path+"/affectus-analysis-model.pickle","rb"))

        predictions = model.predict(vectorizer.transform(texts))

        sentiments = []
        for i in range(len(predictions)):
            sentiment_type = predictions[i]
            sentiment_code = None
            match str(sentiment_type).lower():
                case 'negative':
                    sentiment_code = 1
                case 'positive':
                    sentiment_code = 2
                case 'neutral':
                    sentiment_code = 3
                case 'irrelevant':
                    sentiment_code = 4

            sentiments.append({
                "text": texts[i],
                "code": str(sentiment_code),
                "type": sentiment_type,
            })

        return sentiments