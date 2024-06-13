from nltk.stem.porter import PorterStemmer 
from nltk.corpus import stopwords
import re

def remove_stopwords_then_stem(text : str) -> str :
    porter_stemmer = PorterStemmer() 

    text = re.sub('[^a-zA-Z]',' ', text) 
    text = text.lower() 
    words = text.split() 
    words = [porter_stemmer.stem(word) for word in words if not word in stopwords.words('english')] 
    text = ' '.join(words) 
    return text 
