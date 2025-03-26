from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import os
import joblib
from fastapi import UploadFile, Form
from fastapi.responses import HTMLResponse
from io import StringIO
from fastapi.staticfiles import StaticFiles
from app.custom_preprocessor import SpacyPreprocessor  # Importación necesaria
from pathlib import Path
import pathlib
from typing import Union, Optional
import re

# Parche: cuando el modelo quiera cargar PosixPath, que use WindowsPath
pathlib.PosixPath = pathlib.WindowsPath

app = FastAPI()


app_dir = Path(__file__).resolve().parent
statics_path = app_dir / "statics"
app.mount("/statics", StaticFiles(directory=statics_path), name="statics")

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

@app.get("/", response_class=HTMLResponse)
async def root():
    # Ruta robusta y absoluta al HTML
    base_dir = Path(__file__).resolve().parent
    html_path = base_dir / "templates" / "index.html"
    html = html_path.read_text(encoding="utf-8")
    return HTMLResponse(content=html)

@app.get("/predict", response_class=HTMLResponse)
async def predict_page():
    base_dir = Path(__file__).resolve().parent
    html_path = base_dir / "templates" / "predict.html"
    html = html_path.read_text(encoding="utf-8")
    return HTMLResponse(content=html)

@app.post("/predict", response_class=HTMLResponse)
async def predecir_archivo(
    file: Optional[UploadFile] = Form(default=None),
    titulo: str = Form(default=None),
    descripcion: str = Form(default=None)
    ):
    if file:
        contenido = await file.read()
        df = pd.read_csv(StringIO(contenido.decode('utf-8')), sep=";")

        if "Titulo" not in df.columns or "Descripcion" not in df.columns:
            return HTMLResponse("<h2>❌ Error: El archivo debe tener las columnas 'Titulo' y 'Descripcion'</h2>")

        if df.empty:
            return HTMLResponse("<h2>❌ El archivo está vacío</h2>")

        df_original = df.copy()
        df = df.drop(columns=["Fecha", "ID"], errors="ignore")

    elif titulo and descripcion:
        df_original = pd.DataFrame([{"Titulo": titulo, "Descripcion": descripcion}])
        df = df_original.copy()

    else:
        return HTMLResponse("<h2>❌ Debes subir un archivo CSV o ingresar una noticia manualmente.</h2>")

    modelo = get_model()
    preds = modelo.predict(df)
    probs = modelo.predict_proba(df)[:, 1]

    resultados = []
    for i, (titulo, descripcion, pred, prob) in enumerate(zip(
        df_original["Titulo"], df_original["Descripcion"], preds, probs
    )):
        resultados.append({
            "titulo": titulo,
            "descripcion": descripcion,
            "prediccion": "Verdadera" if pred == 1 else "Falsa",
            "probabilidad": f"{prob:.2f}"
        })

    tabla_filas = ""
    for i, r in enumerate(resultados, 1):
        prob_verdadera = float(r["probabilidad"])
        prob_falsa = 1 - prob_verdadera
        texto = f"""
            {r['titulo']}
            <br>
            <button onclick="toggleDescripcion('desc{i}')">Leer descripción 📄</button>
            <div id="desc{i}" style="display:none; margin-top: 5px; font-style: italic; color: #555;">
                {r['descripcion']}
            </div>
        """
        prob_display = f"""
            <div><strong>Verdadera:</strong> {prob_verdadera:.2f}</div>
            <div><strong>Falsa:</strong> {prob_falsa:.2f}</div>
        """
        tabla_filas += f"""
        <tr>
            <td>{i}</td>
            <td>{texto}</td>
            <td>{r['prediccion']}</td>
            <td>{prob_display}</td>
        </tr>"""

    base_dir = Path(__file__).resolve().parent
    template = (base_dir / "templates" / "predict.html").read_text(encoding="utf-8")
    html_final = template

    # Reemplazo de tabla
    html_final = html_final.replace(
        '<tr><td colspan="4">No hay datos aún</td></tr>', tabla_filas
    )

    # Estadísticas
    num_verdaderas = sum(1 for r in resultados if r["prediccion"] == "Verdadera")
    num_falsas = len(resultados) - num_verdaderas
    total = len(resultados)

    porc_verdaderas = round((num_verdaderas / total) * 100, 1) if total else 0
    porc_falsas = round((num_falsas / total) * 100, 1) if total else 0

    # Reemplazo de cantidades sin comillas
    html_final = html_final.replace('__VERDADERAS__', str(num_verdaderas))
    html_final = html_final.replace('__FALSAS__', str(num_falsas))

    if total > 0:
        html_final = html_final.replace('__', f"{porc_verdaderas}%")
        html_final = html_final.replace('_', f"{porc_falsas}%")
    else:
        html_final = html_final.replace(
        '<p><strong>Verdaderas:</strong> __</p>', ''
        )
        html_final = html_final.replace(
        '<p><strong>Falsas:</strong> _</p>', ''
        )
    return HTMLResponse(content=html_final)


@app.get("/retrain", response_class=HTMLResponse)
async def retrain_page():
    base_dir = Path(__file__).resolve().parent
    html_path = base_dir / "templates" / "retrain.html"
    html = html_path.read_text(encoding="utf-8")
    
    # Reemplazar todos los placeholders con cadenas vacías
    html_final = (
        html.replace("__F1__", "")
            .replace("__RECALL__", "")
            .replace("__PRECISION__", "")
            .replace("__ACCURACY__", "")
            .replace("__TOTAL__", "")
    )
    return HTMLResponse(content=html_final)

@app.post("/retrain", response_class=HTMLResponse)
async def retrain_model(file: UploadFile):
    contenido = await file.read()
    df_nuevo = pd.read_csv(StringIO(contenido.decode('utf-8')), sep=";")

    if not {"Titulo", "Descripcion", "Label"}.issubset(df_nuevo.columns):
        return HTMLResponse("<h2>❌ El archivo debe contener las columnas: Titulo, Descripcion, Label</h2>")

    df_nuevo = df_nuevo.drop(columns=["Fecha", "ID"], errors="ignore")

    
    historial_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'training_data.csv'))

    if os.path.exists(historial_path):
        df_historial = pd.read_csv(historial_path)
        df_total = pd.concat([df_historial, df_nuevo], ignore_index=True).drop_duplicates()
    else:
        df_total = df_nuevo

    df_total.to_csv(historial_path, index=False)

    X = df_total[['Titulo', 'Descripcion']]
    y = df_total['Label']

    modelo = get_model()
    modelo.fit(X, y)

    preds = modelo.predict(X)
    precision = precision_score(y, preds)
    recall = recall_score(y, preds)
    f1 = f1_score(y, preds)

    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'model_bi_project.joblib'))
    joblib.dump(modelo, model_path)
    
    accuracy = accuracy_score(y, preds)
    '''
    precision = 0.56
    recall = 0.33
    f1 = 0.25
    accuracy = 0.89
    df_total = df_nuevo.columns
    '''
    
    base_dir = Path(__file__).resolve().parent
    template = (base_dir / "templates" / "retrain.html").read_text(encoding="utf-8")

    html_final = (
    template.replace("__F1__", f"{f1 * 100:.1f}")
            .replace("__RECALL__", f"{recall * 100:.1f}")
            .replace("__PRECISION__", f"{precision * 100:.1f}")
            .replace("__ACCURACY__", f"{accuracy * 100:.1f}")
            .replace("__TOTAL__", f"{len(df_total)}")
)
    return HTMLResponse(content=html_final)


@app.post("/predecir")
def predecir(datos: list[Noticia]):
    df = pd.DataFrame([d.dict() for d in datos])
    # Eliminar columnas que el modelo no necesita
    df = df.drop(columns=["Fecha", "ID"], errors="ignore")
    modelo = get_model()
    print("antes de predecir")
    preds = modelo.predict(df)
    probs = modelo.predict_proba(df)[:, 1]
    print("después de predecir")
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

