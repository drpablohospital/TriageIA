# =============================
# 1. IMPORTAR LIBRERÍAS
# =============================
import pandas as pd
import numpy as np
from sklearn.metrics import (cohen_kappa_score, classification_report,
                            accuracy_score, precision_score,
                            recall_score, f1_score)
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                         AutoModelForSequenceClassification, pipeline,
                         AutoConfig)
from datasets import Dataset
import torch
from collections import Counter
from tqdm import tqdm
import gradio as gr
import os

# =============================
# 2. CONFIGURACIÓN DE MODELOS Y ETIQUETAS
# =============================
MODELOS_HF = {
    "PRIORIDAD": "drpablo-hospital/modelo_prioridad_old",
    "DERIVACION": "drpablo-hospital/modelo_derivacion",
    "ESPECIALIDAD": "drpablo-hospital/modelo_especialidad",
    "IDX": "drpablo-hospital/modelo_idx_grupo"
}

MODEL_TYPES = {
    "PRIORIDAD": "bert",
    "DERIVACION": "bert",
    "ESPECIALIDAD": "bert",
    "IDX": "t5"
}

# Diccionarios de mapeo de etiquetas
LABEL_MAPPING = {
    "PRIORIDAD": {0: "I", 1: "II", 2: "III", 3: "IV"},
    "DERIVACION": {0: "ALTA", 1: "CONSULTA DE", 2: "INGRESA A"},
    "ESPECIALIDAD": {
        0: "CIRUGIA",
        1: "OFTALMOLOGIA",
        2: "ORTOPEDIA",
        3: "OTORRINOLARINGOLOGIA",
        4: "RECONSTRUCTIVA",
        5: "SIN ESPECIALIDAD",
        6: "URGENCIAS",
        7: "UROLOGIA"
    }
}

# =============================
# 3. FUNCIONES DE PREPROCESAMIENTO (VERSIÓN CORREGIDA)
# =============================
def preprocesar_datos(df):
    df = df.dropna(subset=["MOTIVO", "EDAD", "GENERO", "FC", "FR", "TAS", "SAO2", "TEMP", "NEWS2"])

    # Corrección temporal para TAS (revertir overflow)
    df['TAS'] = df['TAS'].apply(lambda x: x if x >= 0 else 256 + x)

    tipos_optimizados = {
        'EDAD': 'int8',
        'FC': 'int8',
        'FR': 'int8',
        'TAS': 'int16',  # Ahora seguro
        'SAO2': 'int8',
        'TEMP': 'float32',
        'NEWS2': 'int8'
    }
    df = df.astype(tipos_optimizados)

    # Formato IDÉNTICO al usado en entrenamiento
    df["input_text"] = df.apply(lambda x:
        f"[EDAD] {x['EDAD']} [GENERO] {x['GENERO']} [FC] {x['FC']} [FR] {x['FR']} "
        f"[TAS] {x['TAS']} [SAO2] {x['SAO2']} [TEMP] {x['TEMP']} [NEWS2] {x['NEWS2']} "
        f"[MOTIVO] {x['MOTIVO']}", axis=1)

    return df

# =============================
# 4. EVALUACIÓN DE MODELOS (AJUSTES PARA T5)
# =============================
def evaluar_modelos(df):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df_resultados = df.copy()
    resultados = {}
    metricas_detalladas = {}

    for variable, modelo_hf in MODELOS_HF.items():
        try:
            model_type = MODEL_TYPES[variable]
            verdaderos = df[variable].astype(str).tolist()
            textos = df["input_text"].tolist()
            predichos = []

            if model_type == "bert":
                # Configuración para modelos BERT
                tokenizer = AutoTokenizer.from_pretrained(modelo_hf)
                model = AutoModelForSequenceClassification.from_pretrained(modelo_hf).to(device)
                classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

                # Predecir en lotes para mejor rendimiento
                batch_size = 32
                for i in tqdm(range(0, len(textos), batch_size), desc=f"Prediciendo {variable} (BERT)"):
                    batch = textos[i:i+batch_size]
                    preds = classifier(batch)
                    for pred in preds:
                        label = LABEL_MAPPING[variable][int(pred['label'].split('_')[-1])]
                        predichos.append(label)

            elif model_type == "t5":
                # Configuración para modelo T5
                tokenizer = AutoTokenizer.from_pretrained(modelo_hf)
                model = AutoModelForSeq2SeqLM.from_pretrained(modelo_hf).to(device)

                generation_config = {
                    "max_length": 64,
                    "num_beams": 3,
                    "early_stopping": True,
                    "no_repeat_ngram_size": 2,
                    "temperature": 0.7
                }

                for texto in tqdm(textos, desc=f"Prediciendo {variable} (T5)"):
                    inputs = tokenizer(
                        texto,
                        return_tensors="pt",
                        max_length=512,
                        truncation=True,
                        padding="max_length"
                    ).to(device)
                    outputs = model.generate(**inputs, **generation_config)
                    pred = tokenizer.decode(
                        outputs[0],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    ).upper().strip()

                    # Post-procesamiento para IDX
                    if variable == "IDX":
                        pred = pred.replace("  ", " ").replace("..", ".").strip()

                    predichos.append(pred)

            # Filtrar clases no presentes en los datos reales para el reporte
            unique_real = set(verdaderos)
            unique_pred = set(predichos)
            all_labels = sorted(unique_real.union(unique_pred))

            # Calcular métricas
            kappa = cohen_kappa_score(verdaderos, predichos)
            accuracy = accuracy_score(verdaderos, predichos)
            precision = precision_score(verdaderos, predichos, average='weighted', zero_division=0)
            recall = recall_score(verdaderos, predichos, average='weighted', zero_division=0)
            f1 = f1_score(verdaderos, predichos, average='weighted', zero_division=0)

            # Reporte de clasificación
            reporte_str = classification_report(
                verdaderos, predichos,
                labels=all_labels,
                zero_division=0,
                output_dict=False
            )

            # Reporte detallado para archivo TXT
            reporte_dict = classification_report(
                verdaderos, predichos,
                labels=all_labels,
                zero_division=0,
                output_dict=True
            )

            # Guardar resultados
            resultados[variable] = {
                "kappa": kappa,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "reporte": reporte_str,
                "ejemplos": list(zip(verdaderos[:5], predichos[:5]))
            }

            # Guardar métricas detalladas con estructura consistente
            metricas_detalladas[variable] = {
                "metricas_generales": {
                    "kappa": float(kappa),
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1)
                },
                "metricas_por_clase": reporte_dict
            }

            # Agregar predicciones al DataFrame
            df_resultados[f"IA_{variable}"] = predichos

            print(f"✅ {variable} - Kappa: {kappa:.3f}, F1: {f1:.3f}, Accuracy: {accuracy:.3f}")

        except Exception as e:
            print(f"❌ Error evaluando {variable}: {str(e)}")
            resultados[variable] = {"error": str(e)}
            metricas_detalladas[variable] = {
                "error": str(e),
                "metricas_generales": {
                    "kappa": 0.0,
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0
                },
                "metricas_por_clase": {}
            }
            df_resultados[f"IA_{variable}"] = f"Error: {str(e)}"

        # Liberar memoria
        if 'model' in locals():
            del model
        if 'tokenizer' in locals():
            del tokenizer
        if 'classifier' in locals():
            del classifier
        torch.cuda.empty_cache()

    # Guardar resultados en CSV
    output_csv_path = "resultados_modelos.csv"
    df_resultados.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 Resultados CSV guardados en: {output_csv_path}")

    # Guardar métricas detalladas en TXT
    guardar_metricas_txt(metricas_detalladas)

    return resultados, df_resultados

def guardar_metricas_txt(metricas_detalladas):
    """Guarda las métricas detalladas en un archivo TXT"""
    with open("metricas_detalladas.txt", "w", encoding='utf-8') as f:
        f.write("ANÁLISIS DETALLADO DE MODELOS\n")
        f.write("="*50 + "\n\n")

        for variable, datos in metricas_detalladas.items():
            f.write(f"MODELO: {variable.upper()}\n")
            f.write("-"*50 + "\n")

            if "error" in datos:
                f.write(f"ERROR: {datos['error']}\n\n")
                continue

            try:
                # Métricas generales
                f.write("MÉTRICAS GENERALES:\n")
                if isinstance(datos.get("metricas_generales"), dict):
                    for k, v in datos["metricas_generales"].items():
                        if isinstance(v, float):
                            f.write(f"{k}: {v:.4f}\n")
                        else:
                            f.write(f"{k}: {v}\n")
                f.write("\n")

                # Métricas por clase
                f.write("MÉTRICAS POR CLASE:\n")
                if isinstance(datos.get("metricas_por_clase"), dict):
                    for clase, metricas in datos["metricas_por_clase"].items():
                        if clase in ['accuracy', 'macro avg', 'weighted avg']:
                            f.write(f"\n{clase.upper()}:\n")
                            if isinstance(metricas, dict):
                                for k, v in metricas.items():
                                    if isinstance(v, float):
                                        f.write(f"{k}: {v:.4f}\n")
                                    else:
                                        f.write(f"{k}: {v}\n")
                            else:
                                f.write(f"{metricas}\n")
                        elif isinstance(metricas, dict):
                            f.write(f"\nClase {clase}:\n")
                            for k, v in metricas.items():
                                if isinstance(v, float):
                                    f.write(f"{k}: {v:.4f}\n")
                                else:
                                    f.write(f"{k}: {v}\n")
            except Exception as e:
                f.write(f"Error al generar reporte: {str(e)}\n")

            f.write("\n" + "="*50 + "\n\n")

    print(f"💾 Métricas detalladas guardadas en: metricas_detalladas.txt")

# =============================
# 5. FORMATO DE RESULTADOS PARA GRADIO
# =============================
def format_resultados(resultados):
    output = "📊 RESULTADOS FINALES:\n"
    output += "="*60 + "\n"

    for variable, datos in resultados.items():
        output += f"\n🔹 {variable.upper()}\n"
        output += "-"*40 + "\n"

        if "error" in datos:
            output += f"Error: {datos['error']}\n"
            continue

        output += f"Coeficiente Kappa: {datos['kappa']:.3f}\n"
        output += f"Accuracy: {datos['accuracy']:.3f}\n"
        output += f"Precision: {datos['precision']:.3f}\n"
        output += f"Recall: {datos['recall']:.3f}\n"
        output += f"F1-Score: {datos['f1']:.3f}\n"

        output += "\nReporte de Clasificación:\n"
        output += datos["reporte"] + "\n"

        output += "\nEjemplos (Real → Predicho):\n"
        for real, pred in datos["ejemplos"]:
            output += f"- {real} → {pred}\n"

    return output

# =============================
# 6. INTERFAZ GRADIO MEJORADA
# =============================
def cargar_datos(file_obj):
    try:
        df = pd.read_csv(file_obj.name)
        required_columns = ["MOTIVO", "EDAD", "GENERO", "FC", "FR", "TAS", "SAO2", "TEMP", "NEWS2"]

        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            return None, f"❌ Error: Faltan columnas obligatorias: {', '.join(missing)}"

        return df, "✅ Archivo cargado correctamente. Procesando datos..."
    except Exception as e:
        return None, f"❌ Error al cargar archivo: {str(e)}"

def procesar_archivo(file_obj):
    # 1. Cargar datos
    df, mensaje = cargar_datos(file_obj)
    if df is None:
        return mensaje, None, None, None

    # 2. Preprocesar
    df = preprocesar_datos(df)

    # 3. Evaluar modelos
    resultados, df_resultados = evaluar_modelos(df)

    # 4. Formatear resultados
    output = format_resultados(resultados)

    # Crear archivos temporales para descarga
    temp_csv = "temp_resultados.csv"
    df_resultados.to_csv(temp_csv, index=False, encoding='utf-8-sig')

    temp_txt = "temp_metricas.txt"
    with open(temp_txt, "w", encoding='utf-8') as f:
        f.write(output)

    return mensaje, output, temp_csv, temp_txt

# Crear la interfaz
with gr.Blocks() as demo:
    gr.Markdown("## 🔬 Evaluador Avanzado de Modelos Clínicos")
    gr.Markdown("Carga un archivo CSV para evaluar los modelos de clasificación.")

    with gr.Row():
        file_input = gr.File(label="Sube tu archivo CSV", file_types=[".csv"])
        submit_btn = gr.Button("Evaluar Modelos")

    with gr.Row():
        status_output = gr.Textbox(label="Estado", interactive=False)
        results_output = gr.Textbox(label="Resultados", interactive=False, lines=20)

    with gr.Row():
        download_csv = gr.File(label="Descargar resultados completos (CSV)")
        download_txt = gr.File(label="Descargar análisis detallado (TXT)")

    submit_btn.click(
        fn=procesar_archivo,
        inputs=file_input,
        outputs=[status_output, results_output, download_csv, download_txt]
    )

# =============================
# 7. EJECUCIÓN PRINCIPAL
# =============================
if __name__ == "__main__":
    demo.launch()
