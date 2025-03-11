import dill
from sklearn.feature_extraction.text import TfidfVectorizer as SklearnTfidfVectorizer

class TfidfVectorizer:
    def __init__(self, max_df=1.0, min_df=1, ngram_range=(1, 3), tokenizer=lambda x: x.split()):
        self.max_df = max_df
        self.min_df = min_df
        self.ngram_range = ngram_range
        self.vectorizer = None
        self.tokenizerLambda = tokenizer
    
    # Define tokenizer as a method of the class
    def tokenizer(self, text):
        # Your tokenization logic here
        return self.tokenizerLambda(text)
    
    def fit(self, texts):
        self.vectorizer = SklearnTfidfVectorizer(
            max_df=self.max_df,
            min_df=self.min_df,
            ngram_range=self.ngram_range,
            tokenizer=self.tokenizer
        )
        self.vectorizer.fit(texts)
        return self
    
    def transform(self, texts):
        return self.vectorizer.transform(texts)
    
    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)
    
    def get_feature_names_out(self):
        return self.vectorizer.get_feature_names_out()
    
    # Add any other methods you need from TfidfVectorizer
    
    # Method to save the entire class instance
    def save(self, path):
        with open(path, 'wb') as f:
            dill.dump(self, f)
    
    # Class method to load a saved instance
    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return dill.load(f)