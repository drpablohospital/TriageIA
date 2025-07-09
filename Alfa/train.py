# =============================
# 1. IMPORTAR LIBRERÍAS
# =============================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    cohen_kappa_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report
)
from imblearn.over_sampling import RandomOverSampler
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    Trainer, 
    TrainingArguments, 
    TrainerCallback
)
from datasets import Dataset, disable_progress_bar
import tkinter as tk
from tkinter import filedialog
import torch
import gc
from collections import Counter
import os

# Deshabilitar barras de progreso
disable_progress_bar()

# =============================
# 2. CALLBACKS MEJORADOS
# =============================
class KappaEarlyStopping(TrainerCallback):
    def __init__(self, threshold=0.85, patience=2):
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
    
    columnas_necesarias = ["MOTIVO", "EDAD", "GENERO", "FC", "FR", "TAS", "SAO2", "TEMP", "NEWS2", 
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
    df = df.dropna(subset=["MOTIVO", "EDAD", "GENERO", "FC", "FR", "TAS", "SAO2", "TEMP", "NEWS2"])
    
    # Optimizar tipos de datos
    tipos_optimizados = {
        'EDAD': 'int8',
        'FC': 'int8',
        'FR': 'int8',
        'TAS': 'int8',
        'SAO2': 'int8',
        'TEMP': 'float32',
        'NEWS2': 'int8'
    }
    df = df.astype(tipos_optimizados)
    
    # Texto de entrada mejorado pero similar
    df["input_text"] = df.apply(lambda x: 
        f"[MOTIVO] {x['MOTIVO']} [EDAD] {x['EDAD']} [GENERO] {x['GENERO']} "
        f"[FC] {x['FC']} [FR] {x['FR']} [TAS] {x['TAS']} "
        f"[SAO2] {x['SAO2']} [TEMP] {x['TEMP']} [NEWS2] {x['NEWS2']}", axis=1)
    
    return df

# =============================
# 5. BALANCEO DE CLASES
# =============================
def balancear_clases(df, etiqueta):
    print(f"\n⚖️ Balanceando clases para {etiqueta}...")
    counter = Counter(df[etiqueta])
    print(f"Distribución original: {counter}")
    
    # Balanceo más inteligente manteniendo distribución relativa
    ros = RandomOverSampler(sampling_strategy='auto', random_state=42)
    X_res, y_res = ros.fit_resample(df[['input_text']], df[etiqueta])
    
    df_balanced = pd.DataFrame({
        'input_text': X_res['input_text'],
        'target_text': y_res
    })
    
    print(f"Nueva distribución: {Counter(y_res)}")
    return df_balanced

# =============================
# 6. FUNCIÓN DE ENTRENAMIENTO MEJORADA CON MÉTRICAS EXTENDIDAS
# =============================
def entrenar_modelo(df, etiqueta, nombre_modelo):
    print(f"\n🚀 Iniciando entrenamiento para {etiqueta}")
    
    # 1. Balancear datos
    df_balanced = balancear_clases(df, etiqueta)
    
    # 2. Dividir datos
    train_df, test_df = train_test_split(
        df_balanced[["input_text", "target_text"]], 
        test_size=0.2, 
        random_state=42,
        stratify=df_balanced["target_text"]
    )
    
    # 3. Cargar modelo
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # 4. Preparar datasets
    def tokenize_function(examples):
        inputs = tokenizer(
            examples["input_text"],
            max_length=256,
            padding="max_length",
            truncation=True
        )
        
        targets = tokenizer(
            text_target=examples["target_text"],
            max_length=32,
            padding="max_length",
            truncation=True
        )
        
        inputs["labels"] = targets["input_ids"]
        return inputs
    
    train_ds = Dataset.from_pandas(train_df).map(tokenize_function, batched=True)
    test_ds = Dataset.from_pandas(test_df).map(tokenize_function, batched=True)
    
    # 5. Configuración de entrenamiento
    args = TrainingArguments(
        output_dir=f"./{nombre_modelo}_results",
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        learning_rate=3e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        num_train_epochs=10,
        weight_decay=0.01,
        warmup_steps=100,
        logging_dir=f"./{nombre_modelo}_logs",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        report_to="none",
        optim="adamw_torch",
    )
    
    # 6. Función para calcular métricas extendidas
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = np.argmax(preds, axis=1)
        
        # Decodificar predicciones y etiquetas
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Calcular todas las métricas
        kappa = cohen_kappa_score(decoded_labels, decoded_preds)
        accuracy = accuracy_score(decoded_labels, decoded_preds)
        
        # Para métricas que necesitan average (manejar multi-clase)
        average_type = 'micro' if len(set(decoded_labels)) > 2 else 'binary'
        f1 = f1_score(decoded_labels, decoded_preds, average=average_type)
        precision = precision_score(decoded_labels, decoded_preds, average=average_type)
        recall = recall_score(decoded_labels, decoded_preds, average=average_type)
        
        # Reporte de clasificación completo
        report = classification_report(
            decoded_labels, 
            decoded_preds, 
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
    
    # 7. Entrenamiento con callbacks
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
        callbacks=[
            KappaEarlyStopping(threshold=0.7, patience=2),
            TrainerCallback()
        ]
    )
    
    # 8. Entrenar
    trainer.train()
    
    # 9. Evaluación final con métricas extendidas
    eval_results = trainer.evaluate()
    print(f"\n📊 Resultados finales para {etiqueta}:")
    print(f"► Kappa: {eval_results['eval_kappa']:.3f}")
    print(f"► Accuracy: {eval_results['eval_accuracy']:.3f}")
    print(f"► F1-Score: {eval_results['eval_f1']:.3f}")
    print(f"► Precision: {eval_results['eval_precision']:.3f}")
    print(f"► Recall: {eval_results['eval_recall']:.3f}")
    print(f"► Loss: {eval_results['eval_loss']:.3f}")
    
    # Mostrar reporte de clasificación completo
    print("\n📝 Reporte de Clasificación Detallado:")
    report = eval_results['eval_report']
    if isinstance(report, dict):
        # Para clasificación multi-clase
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                print(f"\n🔹 Clase {label}:")
                for k, v in metrics.items():
                    print(f"{k:>12}: {v:.3f}" if isinstance(v, (int, float)) else f"{k:>12}: {v}")
    
    # 10. Guardar modelo
    os.makedirs(nombre_modelo, exist_ok=True)
    trainer.save_model(f"./{nombre_modelo}")
    tokenizer.save_pretrained(f"./{nombre_modelo}")
    
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
            {"etiqueta": "PRIORIDAD", "nombre_modelo": "modelo_gt5_prioridad"},
            # {"etiqueta": "IDX", "nombre_modelo": "modelo_gt5_idx"},
            # {"etiqueta": "DERIVACION", "nombre_modelo": "modelo_gt5_derivacion"},
            # {"etiqueta": "ESPECIALIDAD", "nombre_modelo": "modelo_gt5_especialidad"}
        ]
        
        # Entrenar cada modelo
        resultados_metricas = {}
        for config in config_modelos:
            print(f"\n{'='*60}")
            print(f"🔧 ENTRENANDO MODELO PARA: {config['etiqueta']}")
            print(f"{'='*60}")
            
            metricas = entrenar_modelo(
                df=df,
                etiqueta=config["etiqueta"],
                nombre_modelo=config["nombre_modelo"]
            )
            resultados_metricas[config["etiqueta"]] = metricas
            
            # Limpieza de memoria
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
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