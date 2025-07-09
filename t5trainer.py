# =============================
# 1. IMPORTAR LIBRERÍAS OPTIMIZADAS
# =============================
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    cohen_kappa_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    EarlyStoppingCallback
)
from datasets import Dataset, disable_progress_bar
import tkinter as tk
from tkinter import filedialog
import torch
import gc
from collections import Counter
import os
from torch.utils.data import DataLoader
from transformers import set_seed

# Configuración inicial
set_seed(42)  # Para reproducibilidad
disable_progress_bar()  # Deshabilitar barras de progreso

# =============================
# 2. CALLBACKS MEJORADOS
# =============================
class KappaEarlyStopping(TrainerCallback):
    def __init__(self, threshold=0.9, patience=2):
        self.threshold = threshold
        self.patience = patience
        self.no_improve = 0
        self.best_kappa = -1

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if metrics is None:
            return

        current_kappa = metrics.get("eval_kappa", -1)
        if current_kappa > self.best_kappa:
            self.best_kappa = current_kappa
            self.no_improve = 0
        else:
            self.no_improve += 1

        if current_kappa >= self.threshold:
            print(f"\n🎯 Kappa alcanzado {current_kappa:.3f} >= {self.threshold}. Deteniendo...")
            control.should_training_stop = True
        elif self.no_improve >= self.patience:
            print(f"\n⏹️ Sin mejora en Kappa por {self.patience} evaluaciones. Deteniendo...")
            control.should_training_stop = True

# =============================
# 3. CARGA DE DATOS
# =============================
def cargar_datos():
    print("🔼 Selecciona el archivo CSV con tus datos")
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Selecciona el archivo CSV", filetypes=[("CSV Files", "*.csv")])

    columnas_necesarias = ["MOTIVO", "EDAD", "GENERO", "FC", "FR", "TAS", "TAD", "SAO2", "TEMP", "NEWS2",
                          "PRIORIDAD", "IDX", "DERIVACION", "ESPECIALIDAD"]

    df = pd.read_csv(file_path, usecols=columnas_necesarias)
    print(f"✅ Archivo cargado: {file_path}")

    # Mostrar distribución de clases
    print("\n📊 Distribución inicial de clases:")
    for col in ["PRIORIDAD", "DERIVACION", "ESPECIALIDAD", "IDX"]:
        if col in df:
            print(f"{col}: {dict(Counter(df[col].dropna()))}")

    return df

# =============================
# 4. PREPROCESAMIENTO MEJORADO
# =============================
def preprocesar_datos(df):
    # Limpieza manteniendo tus reglas originales
    df = df.dropna(subset=["MOTIVO", "EDAD", "GENERO", "FC", "FR", "TAS", "TAD", "SAO2", "TEMP", "NEWS2"])

    # Optimizar tipos de datos
    tipos_optimizados = {
        'EDAD': 'int8',
        'FC': 'int16',
        'FR': 'int8',
        'TAS': 'int16',
        'TAD': 'TAD',
        'SAO2': 'int8',
        'TEMP': 'float32',
        'NEWS2': 'int8'
    }
    df = df.astype(tipos_optimizados)

    # Texto de entrada modificado: datos estructurados primero, motivo al final
    df["input_text"] = df.apply(lambda x:
        f"[EDAD] {x['EDAD']} [GENERO] {x['GENERO']} [FC] {x['FC']} [FR] {x['FR']} "
        f"[TAS] {x['TAS']} [TAD] {x['TAD']} [SAO2] {x['SAO2']} [TEMP] {x['TEMP']} [NEWS2] {x['NEWS2']} "
        f"[MOTIVO] {x['MOTIVO']}", axis=1)

    return df

# =============================
# 5. BALANCEO INTELIGENTE DE CLASES (VERSIÓN OPTIMIZADA)
# =============================
def balancear_clases(df, etiqueta, target_samples=146:
    print(f"\n⚖️ Balanceando clases para {etiqueta}...")
    counter = Counter(df[etiqueta])
    print(f"Distribución original: {counter}")

    # Convertir etiquetas a numéricas
    unique_labels = sorted(df[etiqueta].unique())
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    df['label_numeric'] = df[etiqueta].map(label_to_id)

    # Estrategia simplificada y más rápida
    dfs = []
    for label, count in counter.items():
        label_df = df[df[etiqueta] == label]

        # Muestreo con reemplazo para clases minoritarias
        if count < target_samples:
            resampled_df = label_df.sample(target_samples, replace=True, random_state=42)
        else:
            # Muestreo sin reemplazo para clases mayoritarias
            resampled_df = label_df.sample(target_samples, random_state=42)

        dfs.append(resampled_df[['input_text', 'label_numeric']].rename(columns={'label_numeric': 'label'}))

    df_balanced = pd.concat(dfs)
    df_balanced['label_text'] = [unique_labels[idx] for idx in df_balanced['label']]

    print(f"Nueva distribución ({len(df_balanced)} muestras): {Counter(df_balanced['label_text'])}")
    return df_balanced

# =============================
# 6. FUNCIÓN DE ENTRENAMIENTO PARA T5 (NUEVA VERSIÓN)
# =============================
def entrenar_modelo_t5(df, etiqueta, nombre_modelo):
    print(f"\n🚀 Iniciando entrenamiento para {etiqueta} con T5")

    # 1. Balancear datos
    df_balanced = balancear_clases(df, etiqueta, target_samples=146)

    # 2. Dividir datos
    train_df, test_df = train_test_split(
        df_balanced[["input_text", "label_text"]],  # Usamos label_text en lugar de label numérico
        test_size=0.2,
        random_state=42,
        stratify=df_balanced["label"]
    )

    # 3. Cargar modelo y tokenizer de T5
    model_name = "t5-small"  # Puedes usar "t5-base" para mejor rendimiento si tienes recursos
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Convertir a formato text-to-text
    train_df['input'] = "clasificar: " + train_df['input_text']
    train_df['target'] = train_df['label_text']
    test_df['input'] = "clasificar: " + test_df['input_text']
    test_df['target'] = test_df['label_text']

    # 4. Preparar datasets
    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples["input"],
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Tokenizar targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["target"],
                max_length=32,  # Más corto para las etiquetas
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_ds = Dataset.from_pandas(train_df).map(
        tokenize_function,
        batched=True,
        batch_size=32,
        remove_columns=['input_text', 'label_text', 'input', 'target']
    )

    test_ds = Dataset.from_pandas(test_df).map(
        tokenize_function,
        batched=True,
        batch_size=32,
        remove_columns=['input_text', 'label_text', 'input', 'target']
    )

    # 5. Cargar modelo T5 para sequence-to-sequence
    from transformers import T5ForConditionalGeneration
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # 6. Configuración de entrenamiento (ajustada para T5)
    args = TrainingArguments(
        output_dir=f"./{nombre_modelo}_results",
        eval_strategy="steps",  # Cambiado de evaluation_strategy
        eval_steps=50,
        save_strategy="steps",  # Cambiado de save_strategy
        save_steps=50,
        learning_rate=2e-5,
        per_device_train_batch_size=16 if torch.cuda.is_available() else 8,
        per_device_eval_batch_size=32 if torch.cuda.is_available() else 16,
        num_train_epochs=5,
        weight_decay=0.01,
        warmup_steps=50,
        logging_dir=f"./{nombre_modelo}_logs",
        load_best_model_at_end=True,
        metric_for_best_model="eval_kappa",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
        optim="adamw_torch",
        logging_steps=50,
        dataloader_num_workers=4 if torch.cuda.is_available() else 0,
        gradient_accumulation_steps=2 if torch.cuda.is_available() else 1,
        seed=42
    )

    # 7. Función para calcular métricas extendidas (adaptada para T5)
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Reemplazar -100 en labels (ignorados por la loss)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Convertir a numéricos para métricas
        unique_labels = sorted(df[etiqueta].unique())
        label_to_id = {label: idx for idx, label in enumerate(unique_labels)}

        preds_numeric = [label_to_id.get(pred, 0) for pred in decoded_preds]
        labels_numeric = [label_to_id.get(label, 0) for label in decoded_labels]

        # Calcular métricas
        kappa = cohen_kappa_score(labels_numeric, preds_numeric)
        accuracy = accuracy_score(labels_numeric, preds_numeric)
        f1 = f1_score(labels_numeric, preds_numeric, average='weighted')
        precision = precision_score(labels_numeric, preds_numeric, average='weighted')
        recall = recall_score(labels_numeric, preds_numeric, average='weighted')

        report = classification_report(
            labels_numeric,
            preds_numeric,
            output_dict=True,
            zero_division=0
        )

        return {
            "kappa": kappa,
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "report": report
        }

    # 8. Entrenamiento con callbacks
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
        callbacks=[
            KappaEarlyStopping(threshold=0.7, patience=2),  # Umbral más bajo para IDX
            EarlyStoppingCallback(early_stopping_patience=3)
        ]
    )

    # 9. Entrenar
    print("\n⏳ Iniciando entrenamiento con T5...")
    trainer.train()

    # 10. Evaluación final
    eval_results = trainer.evaluate()
    print(f"\n📊 Resultados finales para {etiqueta}:")
    print(f"► Kappa: {eval_results['eval_kappa']:.3f}")
    print(f"► Accuracy: {eval_results['eval_accuracy']:.3f}")
    print(f"► F1-Score: {eval_results['eval_f1']:.3f}")
    print(f"► Precision: {eval_results['eval_precision']:.3f}")
    print(f"► Recall: {eval_results['eval_recall']:.3f}")
    print(f"► Loss: {eval_results['eval_loss']:.3f}")

    # Mostrar reporte de clasificación
    print("\n📝 Reporte de Clasificación Detallado:")
    report = eval_results['eval_report']
    if isinstance(report, dict):
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                print(f"\n🔹 Clase {label}:")
                for k, v in metrics.items():
                    print(f"{k:>12}: {v:.3f}" if isinstance(v, (int, float)) else f"{k:>12}: {v}")

    # 11. Guardar modelo
    os.makedirs(nombre_modelo, exist_ok=True)
    trainer.save_model(f"./{nombre_modelo}")
    tokenizer.save_pretrained(f"./{nombre_modelo}")

    # Guardar mapeo de etiquetas
    unique_labels = sorted(df[etiqueta].unique())
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    with open(f"{nombre_modelo}/label_mapping.json", "w") as f:
        json.dump({
            "id2label": {v: k for k, v in label_to_id.items()},
            "label2id": label_to_id
        }, f)

    return {
        "kappa": eval_results['eval_kappa'],
        "accuracy": eval_results['eval_accuracy'],
        "f1": eval_results['eval_f1'],
        "precision": eval_results['eval_precision'],
        "recall": eval_results['eval_recall'],
        "loss": eval_results['eval_loss']
    }

# =============================
# 7. EJECUCIÓN PRINCIPAL
# =============================
if __name__ == "__main__":
    try:
        # Cargar y preparar datos
        df = cargar_datos()
        df = preprocesar_datos(df)

        # Configuración de modelos
        config_modelos = [
            #{"etiqueta": "PRIORIDAD", "nombre_modelo": "modelo_t5_prioridad"},
            {"etiqueta": "IDX", "nombre_modelo": "modelo_t5_idx"},
            #{"etiqueta": "DERIVACION", "nombre_modelo": "modelo_t5_derivacion"},
            #{"etiqueta": "ESPECIALIDAD", "nombre_modelo": "modelo_t5_especialidad"}
        ]

        # Entrenar cada modelo
        resultados_metricas = {}
        for config in config_modelos:
            print(f"\n{'='*60}")
            print(f"🔧 ENTRENANDO MODELO PARA: {config['etiqueta']}")
            print(f"{'='*60}")

            metricas = entrenar_modelo_t5(
                df=df,
                etiqueta=config["etiqueta"],
                nombre_modelo=config["nombre_modelo"]
            )
            resultados_metricas[config["etiqueta"]] = metricas

            # Limpieza de memoria
            torch.cuda.empty_cache()
            gc.collect()

        # Resumen final mejorado
        print("\n📌 RESUMEN FINAL DE MÉTRICAS:")
        print(f"{'Etiqueta':<12} | {'Kappa':<6} | {'Accuracy':<8} | {'F1':<6} | {'Precision':<9} | {'Recall':<6} | {'Loss':<6}")
        print("-"*80)
        for etiqueta, metricas in resultados_metricas.items():
            print(
                f"{etiqueta:<12} | {metricas['kappa']:.3f} | {metricas['accuracy']:.3f}    | "
                f"{metricas['f1']:.3f} | {metricas['precision']:.3f}     | "
                f"{metricas['recall']:.3f} | {metricas['loss']:.3f}"
            )

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
