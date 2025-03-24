import spacy
import re, unicodedata
from sklearn.base import TransformerMixin, BaseEstimator

class SpacyPreprocessor(TransformerMixin, BaseEstimator):
    def __init__(self, lang='es_core_news_sm'):
        self.lang = lang
        self.nlp = spacy.load(lang, disable=['parser', 'ner'])
        self.stop_words = set(spacy.lang.es.stop_words.STOP_WORDS)

    def preprocessing(self, words):
        words = [word.lower() for word in words]
        words = [re.sub(r'[^\w\s]', '', word) for word in words if word]
        words = [unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore') for word in words]
        words = [word for word in words if word not in self.stop_words and word.strip() != ""]
        return " ".join(words)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        textos_procesados = []
        for doc in self.nlp.pipe(X.astype(str).fillna(""), batch_size=100):
            tokens = [token.lemma_ for token in doc if not token.is_space]
            texto_limpio = self.preprocessing(tokens)
            textos_procesados.append(texto_limpio)
        return textos_procesados
