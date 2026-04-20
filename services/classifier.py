import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class DocumentClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.model = LogisticRegression()

    def train(self, texts, labels):
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)

    # def predict(self, text):
    #     X = self.vectorizer.transform([text])
    #     return self.model.predict(X)[0]

    def save(self, path="model.pkl"):
        joblib.dump((self.vectorizer, self.model), path)

    def load(self, path="model.pkl"):
        self.vectorizer, self.model = joblib.load(path)

    def predict_with_confidence(self, text):
        X = self.vectorizer.transform([text])
        probs = self.model.predict_proba(X)

        predicted_class = self.model.classes_[probs.argmax()]
        confidence = probs.max()

        return predicted_class, confidence
