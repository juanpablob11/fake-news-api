from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import joblib

from app.custom_preprocessor import SpacyPreprocessor  # Importación necesaria

app = FastAPI()

from typing import Union, Optional

class Noticia(BaseModel):
    ID: Optional[Union[int, str]] = None
    Titulo: str
    Descripcion: str
    Fecha: str

class NoticiaTrain(Noticia):
    Label: int

model = None

def get_model():
    global model
    if model is None:
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'model_bi_project.joblib'))
        with open(model_path, "rb") as f:
            model = joblib.load(f)
    return model

@app.post("/predecir")
def predecir(datos: list[Noticia]):
    df = pd.DataFrame([d.dict() for d in datos])
    # Eliminar columnas que el modelo no necesita
    df = df.drop(columns=["Fecha", "ID"], errors="ignore")
    modelo = get_model()
    preds = modelo.predict(df)
    probs = modelo.predict_proba(df)[:, 1]
    return [{"prediccion": int(pred), "probabilidad": float(prob)} for pred, prob in zip(preds, probs)]

@app.post("/reentrenar")
def reentrenar(datos: list[NoticiaTrain]):
    # Convertir nuevos datos en DataFrame
    df_nuevo = pd.DataFrame([d.dict() for d in datos])
    df_nuevo = df_nuevo.drop(columns=["Fecha", "ID"], errors="ignore")

    # Ruta del historial acumulado
    historial_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'training_data.csv'))

    # Cargar el histórico si existe
    if os.path.exists(historial_path):
        df_historial = pd.read_csv(historial_path)
        # Evitar duplicados exactos
        df_total = pd.concat([df_historial, df_nuevo], ignore_index=True).drop_duplicates()
    else:
        df_total = df_nuevo

    # Guardar el historial actualizado
    df_total.to_csv(historial_path, index=False)

    # Entrenar modelo desde cero con todo
    X = df_total[['Titulo', 'Descripcion']]
    y = df_total['Label']

    modelo = get_model()
    modelo.fit(X, y)

    # Métricas básicas
    preds = modelo.predict(X)
    precision = precision_score(y, preds)
    recall = recall_score(y, preds)
    f1 = f1_score(y, preds)

    # Guardar modelo actualizado
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'model_bi_project.joblib'))
    joblib.dump(modelo, model_path)

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "registros_entrenados": len(df_total),
        "mensaje": "Modelo reentrenado acumulando datos anteriores."
    }

