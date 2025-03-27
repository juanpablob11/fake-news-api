import spacy
import re, unicodedata
from sklearn.base import TransformerMixin, BaseEstimator

# Clase que implementa un preprocesador de texto usando spaCy para limpieza y lematización.
class SpacyPreprocessor(TransformerMixin, BaseEstimator):
    
    def __init__(self, lang='es_core_news_sm'):
        """
        Inicializa el preprocesador, cargando el modelo de spaCy en el idioma especificado
        (por defecto, 'es_core_news_sm' para español).
        
        Args:
            lang (str): El modelo de idioma de spaCy que se va a usar para el preprocesamiento.
        """
        # Asigna el modelo de idioma a la variable `lang`
        self.lang = lang
        # Carga el modelo de spaCy sin las componentes de parser y named entity recognition (NER)
        self.nlp = spacy.load(lang, disable=['parser', 'ner'])
        # Carga el conjunto de stopwords (palabras vacías) del idioma español
        self.stop_words = set(spacy.lang.es.stop_words.STOP_WORDS)

    def preprocessing(self, words):
        """
        Aplica una serie de transformaciones a una lista de palabras, que incluyen:
        - Conversión a minúsculas.
        - Eliminación de caracteres no alfabéticos (como puntuación).
        - Normalización de caracteres Unicode a su forma ASCII.
        - Eliminación de palabras vacías y espacios adicionales.
        
        Args:
            words (list): Lista de palabras a preprocesar.
        
        Returns:
            str: Texto procesado y concatenado en una sola cadena.
        """
        # Convertir todas las palabras a minúsculas
        words = [word.lower() for word in words]
        
        # Eliminar signos de puntuación de cada palabra
        words = [re.sub(r'[^\w\s]', '', word) for word in words if word]
        
        # Normalizar los caracteres unicode a ASCII (eliminando caracteres especiales)
        words = [unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore') for word in words]
        
        # Eliminar palabras vacías (stopwords) y palabras con solo espacios
        words = [word for word in words if word not in self.stop_words and word.strip() != ""]
        
        # Devolver las palabras procesadas como una cadena de texto
        return " ".join(words)

    def fit(self, X, y=None):
        """
        Método que se utiliza en el pipeline de Scikit-learn. No hace nada en este caso,
        ya que no es necesario aprender de los datos para este transformador.
        
        Args:
            X (array-like): Los datos de entrada (no se utilizan aquí).
            y (array-like, optional): Las etiquetas (no se utilizan aquí).
        
        Returns:
            self: El propio objeto para encadenar transformaciones.
        """
        return self

    def transform(self, X, y=None):
        """
        Realiza la transformación de un conjunto de datos de texto (X), aplicando el modelo spaCy
        para lematizar cada palabra y luego aplicando el preprocesamiento personalizado.
        
        Args:
            X (array-like): Un conjunto de textos a procesar.
            y (array-like, optional): Las etiquetas (no se utilizan aquí).
        
        Returns:
            list: Una lista de textos preprocesados.
        """
        # Lista que almacenará los textos procesados
        textos_procesados = []
        
        # Procesar los textos en lotes (batch_size=100) para mayor eficiencia
        for doc in self.nlp.pipe(X.astype(str).fillna(""), batch_size=100):
            # Lematizar cada token del texto, eliminando los espacios
            tokens = [token.lemma_ for token in doc if not token.is_space]
            
            # Preprocesar los tokens lematizados
            texto_limpio = self.preprocessing(tokens)
            
            # Agregar el texto preprocesado a la lista
            textos_procesados.append(texto_limpio)
        
        # Devolver la lista de textos procesados
        return textos_procesados
