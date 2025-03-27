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

# Directorio de la aplicación
app_dir = Path(__file__).resolve().parent
statics_path = app_dir / "statics"
app.mount("/statics", StaticFiles(directory=statics_path), name="statics")

# Modelos de datos
class Noticia(BaseModel):
    ID: Optional[Union[int, str]] = None
    Titulo: str
    Descripcion: str
    Fecha: str

class NoticiaTrain(Noticia):
    Label: int

# Modelo global
model = None

def get_model():
    """
    Carga el modelo desde un archivo `.joblib` si no ha sido cargado previamente.
    El modelo es almacenado en una variable global para su reutilización.

    Returns:
        model: El modelo de predicción cargado.
    """
    global model
    if model is None:
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'model_bi_project.joblib'))
        with open(model_path, "rb") as f:
            model = joblib.load(f)
    return model

@app.get("/", response_class=HTMLResponse)
async def root():
    """
    Ruta principal que sirve la página HTML `index.html`.
    """
    base_dir = Path(__file__).resolve().parent
    html_path = base_dir / "templates" / "index.html"
    html = html_path.read_text(encoding="utf-8")
    return HTMLResponse(content=html)

@app.get("/predict", response_class=HTMLResponse)
async def predict_page():
    """
    Ruta que sirve la página HTML `predict.html` para la predicción de noticias.
    """
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
    """
    Ruta para realizar la predicción de noticias. Acepta un archivo CSV o noticias manualmente ingresadas
    y muestra los resultados de la predicción junto con la probabilidad.

    Parámetros:
        file: Archivo CSV con las columnas 'Titulo' y 'Descripcion'.
        titulo: Titulo de la noticia ingresada manualmente.
        descripcion: Descripción de la noticia ingresada manualmente.

    Returns:
        HTMLResponse: Contenido HTML con la tabla de predicciones y sus probabilidades.
    """
    if file:
        # Leer el archivo CSV y convertirlo en un DataFrame
        contenido = await file.read()
        df = pd.read_csv(StringIO(contenido.decode('utf-8')), sep=";")

        # Verificar que las columnas 'Titulo' y 'Descripcion' estén presentes
        if "Titulo" not in df.columns or "Descripcion" not in df.columns:
            return HTMLResponse("<h2>❌ Error: El archivo debe tener las columnas 'Titulo' y 'Descripcion'</h2>")

        # Verificar que el archivo no esté vacío
        if df.empty:
            return HTMLResponse("<h2>❌ El archivo está vacío</h2>")

        # Guardar una copia original y eliminar columnas innecesarias
        df_original = df.copy()
        df = df.drop(columns=["Fecha", "ID"], errors="ignore")

    elif titulo and descripcion:
        # Si se ingresan datos manualmente, crear un DataFrame con los datos
        df_original = pd.DataFrame([{"Titulo": titulo, "Descripcion": descripcion}])
        df = df_original.copy()

    else:
        return HTMLResponse("<h2>❌ Debes subir un archivo CSV o ingresar una noticia manualmente.</h2>")

    # Obtener el modelo y hacer las predicciones
    modelo = get_model()
    preds = modelo.predict(df)
    probs = modelo.predict_proba(df)[:, 1]

    # Preparar los resultados para la tabla de HTML
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

    # Generar las filas de la tabla HTML con los resultados
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

    # Obtener y reemplazar la tabla de resultados en la página HTML
    base_dir = Path(__file__).resolve().parent
    template = (base_dir / "templates" / "predict.html").read_text(encoding="utf-8")
    html_final = template

    html_final = html_final.replace(
        '<tr><td colspan="4">No hay datos aún</td></tr>', tabla_filas
    )

    # Calcular estadísticas de las predicciones
    num_verdaderas = sum(1 for r in resultados if r["prediccion"] == "Verdadera")
    num_falsas = len(resultados) - num_verdaderas
    total = len(resultados)

    porc_verdaderas = round((num_verdaderas / total) * 100, 1) if total else 0
    porc_falsas = round((num_falsas / total) * 100, 1) if total else 0

    # Reemplazar placeholders con las estadísticas
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

    # Devolver el HTML con las predicciones y estadísticas
    return HTMLResponse(content=html_final)

@app.get("/retrain", response_class=HTMLResponse)
async def retrain_page():
    """
    Muestra la página de reentrenamiento del modelo con estadísticas de desempeño.
    """
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
    """
    Ruta que permite reentrenar el modelo con nuevos datos. Acepta un archivo CSV con las columnas
    'Titulo', 'Descripcion' y 'Label', y actualiza el modelo.

    Parámetros:
        file: Archivo CSV con las columnas 'Titulo', 'Descripcion' y 'Label'.

    Returns:
        HTMLResponse: Contenido HTML con las métricas del modelo reentrenado.
    """
    contenido = await file.read()
    df_nuevo = pd.read_csv(StringIO(contenido.decode('utf-8')), sep=";")

    # Verificar que el archivo contenga las columnas necesarias
    if not {"Titulo", "Descripcion", "Label"}.issubset(df_nuevo.columns):
        return HTMLResponse("<h2>❌ El archivo debe contener las columnas: Titulo, Descripcion, Label</h2>")

    df_nuevo = df_nuevo.drop(columns=["Fecha", "ID"], errors="ignore")

    # Ruta para almacenar los datos históricos de entrenamiento
    historial_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'training_data.csv'))

    # Cargar datos históricos y agregar los nuevos
    if os.path.exists(historial_path):
        df_historial = pd.read_csv(historial_path)
        df_total = pd.concat([df_historial, df_nuevo], ignore_index=True).drop_duplicates()
    else:
        df_total = df_nuevo

    # Guardar los datos combinados
    df_total.to_csv(historial_path, index=False)

    # Separar las características y las etiquetas
    X = df_total[['Titulo', 'Descripcion']]
    y = df_total['Label']

    # Obtener el modelo y reentrenarlo
    modelo = get_model()
    modelo.fit(X, y)

    # Calcular métricas del modelo
    preds = modelo.predict(X)
    precision = precision_score(y, preds)
    recall = recall_score(y, preds)
    f1 = f1_score(y, preds)

    # Guardar el modelo actualizado
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'model_bi_project.joblib'))
    joblib.dump(modelo, model_path)
    
    # Calcular la precisión general del modelo
    accuracy = accuracy_score(y, preds)

    # Preparar el HTML para mostrar las métricas
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
    """
    Realiza una predicción sobre una lista de noticias proporcionada como entrada.

    Parámetros:
        datos: Lista de objetos `Noticia` que contienen el título y descripción de cada noticia.

    Returns:
        list: Lista con las predicciones y probabilidades de cada noticia.
    """
    df = pd.DataFrame([d.dict() for d in datos])
    df = df.drop(columns=["Fecha", "ID"], errors="ignore")

    modelo = get_model()

    # Realizar las predicciones y calcular las probabilidades
    preds = modelo.predict(df)
    probs = modelo.predict_proba(df)[:, 1]

    # Retornar las predicciones y probabilidades
    return [{"prediccion": int(pred), "probabilidad": float(prob)} for pred, prob in zip(preds, probs)]

@app.post("/reentrenar")
def reentrenar(datos: list[NoticiaTrain]):
    """
    Reentrena el modelo con nuevos datos de entrenamiento. Los datos son recibidos como una lista de objetos `NoticiaTrain`.

    Parámetros:
        datos: Lista de objetos `NoticiaTrain` que contienen título, descripción y etiqueta (label).

    Returns:
        dict: Diccionario con las métricas del modelo y el número de registros entrenados.
    """
    df_nuevo = pd.DataFrame([d.dict() for d in datos])
    df_nuevo = df_nuevo.drop(columns=["Fecha", "ID"], errors="ignore")

    # Ruta del archivo que contiene los datos históricos de entrenamiento
    historial_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'training_data.csv'))

    # Si existe el archivo histórico, combinar los datos nuevos y antiguos
    if os.path.exists(historial_path):
        df_historial = pd.read_csv(historial_path)
        df_total = pd.concat([df_historial, df_nuevo], ignore_index=True).drop_duplicates()
    else:
        df_total = df_nuevo

    # Guardar el archivo histórico actualizado
    df_total.to_csv(historial_path, index=False)

    # Entrenar el modelo con todos los datos disponibles
    X = df_total[['Titulo', 'Descripcion']]
    y = df_total['Label']

    modelo = get_model()
    modelo.fit(X, y)

    # Calcular las métricas del modelo
    preds = modelo.predict(X)
    precision = precision_score(y, preds)
    recall = recall_score(y, preds)
    f1 = f1_score(y, preds)

    # Guardar el modelo actualizado
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'model_bi_project.joblib'))
    joblib.dump(modelo, model_path)

    # Retornar las métricas y el número de registros entrenados
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "registros_entrenados": len(df_total),
        "mensaje": "Modelo reentrenado acumulando datos anteriores."
    }
