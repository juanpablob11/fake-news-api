import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from app.custom_preprocessor import SpacyPreprocessor  

# Cargar dataset
df = pd.read_csv('./data/Noticias_Falsas_Extendidas_50-50.csv').dropna(subset=['Titulo', 'Descripcion']).drop_duplicates()

# Eliminar duplicados parciales con diferentes etiquetas
duplicated_mask = df.duplicated(subset=['Titulo', 'Descripcion'], keep=False)
duplicados = df[duplicated_mask]
duplicados_grouped = duplicados.groupby(['Titulo', 'Descripcion']).filter(lambda x: x['Label'].nunique() > 1)
df = df.drop(duplicados_grouped.index)

# Seleccionar 4000 noticias falsas y 4000 verdaderas para testeo externo
test_df_0 = df[df['Label'] == 0].sample(n=4000, random_state=42)
test_df_1 = df[df['Label'] == 1].sample(n=4000, random_state=42)
external_test_df = pd.concat([test_df_0, test_df_1]).sample(frac=1, random_state=42)  # Mezclar

# Guardar como CSV para evaluación externa
external_test_df.to_csv('./data/test_dataset.csv', index=False)
print("Dataset externo de evaluación guardado en: ./data/test_dataset.csv")

# Remover estos registros del dataset original
df = df.drop(external_test_df.index)

# Variables finales (sin 'Fecha')
X = df[['Titulo', 'Descripcion']]
y = df['Label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Column Transformer sin fecha
transformer = ColumnTransformer([
    ('titulo_pipeline', Pipeline([
        ('preprocess', SpacyPreprocessor()),
        ('tfidf', TfidfVectorizer(ngram_range=(1, 3), max_features=6000))
    ]), 'Titulo'),

    ('descripcion_pipeline', Pipeline([
        ('preprocess', SpacyPreprocessor()),
        ('tfidf', TfidfVectorizer(ngram_range=(1, 3), max_features=6000))
    ]), 'Descripcion')
])

# Pipeline completo
pipeline = Pipeline([
    ('vectorizacion', transformer),
    ('clasificador', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Entrenar modelo
pipeline.fit(X_train, y_train)

# Guardar modelo con joblib
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'model_bi_project.joblib'))
joblib.dump(pipeline, model_path)

print("Modelo entrenado y guardado exitosamente en:", model_path)

# Guardar dataset de entrenamiento original usado (para futuros reentrenamientos)
entrenamiento_df = X.copy()
entrenamiento_df['Label'] = y
entrenamiento_df.to_csv('./data/training_data.csv', index=False)
print("Datos originales de entrenamiento guardados en: ./data/training_data.csv")
