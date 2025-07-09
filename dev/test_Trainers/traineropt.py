# =============================
# 1. IMPORTAR LIBRERÍAS (OPTIMIZADO)
# =============================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, TrainerCallback
from datasets import Dataset, disable_progress_bar
import tkinter as tk
from tkinter import filedialog
import torch
import gc

# Deshabilitar barras de progreso para reducir overhead
disable_progress_bar()

# =============================
# 2. DEFINIR CUSTOM CALLBACK (OPTIMIZADO)
# =============================
class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, threshold, patience=2):
        self.threshold = threshold
        self.patience = patience
        self.no_improve = 0
        self.best_loss = float('inf')
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs:
            current_loss = logs['loss']
            if current_loss < self.threshold:
                print(f"\n⚡ Loss {current_loss:.4f} < {self.threshold}. Deteniendo...")
                control.should_training_stop = True
            
            # Lógica de paciencia para evitar paradas prematuras
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.no_improve = 0
            else:
                self.no_improve += 1
                if self.no_improve >= self.patience:
                    print(f"\n⏹️ Sin mejora por {self.patience} epochs. Deteniendo...")
                    control.should_training_stop = True

# =============================
# 3. CARGA DE DATOS (OPTIMIZADO)
# =============================
def cargar_datos():
    print("🔼 Selecciona el archivo CSV con tus datos")
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Selecciona el archivo CSV", filetypes=[("CSV Files", "*.csv")])
    
    # Cargar solo columnas necesarias para ahorrar memoria
    columnas_necesarias = ["MOTIVO", "EDAD", "GENERO", "FC", "FR", "TAS", "SAO2", "TEMP", "NEWS2", 
                          "PRIORIDAD", "IDX", "DERIVACION", "ESPECIALIDAD"]
    
    df = pd.read_csv(file_path, usecols=columnas_necesarias)
    print(f"✅ Archivo cargado: {file_path}")
    return df

df = cargar_datos()

# =============================
# 4. PREPROCESAMIENTO (OPTIMIZADO)
# =============================
def preprocesar_datos(df):
    # Eliminar NA y convertir a tipos óptimos
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
    
    # Construir texto de entrada (más eficiente con f-strings)
    df["input_text"] = df.apply(lambda x: 
        f"Motivo: {x['MOTIVO']} | Edad: {x['EDAD']} | Genero: {x['GENERO']} | "
        f"FC: {x['FC']} | FR: {x['FR']} | TAS: {x['TAS']} | "
        f"SAO2: {x['SAO2']} | TEMP: {x['TEMP']} | NEWS2: {x['NEWS2']}", axis=1)
    
    return df

df = preprocesar_datos(df)

# =============================
# 5. FUNCIÓN DE ENTRENAMIENTO (CORREGIDA Y OPTIMIZADA)
# =============================
def entrenar_t5(df, etiqueta, nombre_modelo, balancear=True, early_stopping_threshold=0.03):
    print(f"\n🚀 Entrenando T5 para: {etiqueta} (Optimizado para CPU)")
    
    # Configuración para bajo consumo de memoria
    torch.set_num_threads(2)  # Limitar hilos de CPU
    torch.set_flush_denormal(True)  # Optimización matemática
    
    # 1. Preparación de datos
    df_temp = df.dropna(subset=[etiqueta]).copy()
    df_temp["target_text"] = df_temp[etiqueta].astype('category')  # Más eficiente que str
    
    if balancear:
        print("⚖️ Aplicando balanceo optimizado...")
        # Balanceo con muestreo estratificado más eficiente
        class_counts = df_temp["target_text"].value_counts()
        min_size = class_counts.min()
        df_balanced = pd.concat([
            df_temp[df_temp["target_text"] == cls].sample(min_size, replace=True, random_state=42)
            for cls in class_counts.index
        ])
    else:
        df_balanced = df_temp
    
    # 2. Crear datasets optimizados
    train_df, test_df = train_test_split(
        df_balanced[["input_text", "target_text"]], 
        test_size=0.1, 
        random_state=42, 
        stratify=df_balanced["target_text"]
    )
    
    # Liberar memoria
    del df_temp, df_balanced
    gc.collect()
    
    # 3. Tokenización CORREGIDA para modelo seq2seq
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    
    def preprocess_function(examples):
        # Tokenizar inputs (encoder)
        inputs = tokenizer(
            examples["input_text"],
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="np"  # Mantenemos numpy para eficiencia en CPU
        )
        
        # Tokenizar targets (decoder)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["target_text"],
                max_length=32,
                padding="max_length",
                truncation=True,
                return_tensors="np"  # Mantenemos numpy para eficiencia
            )
        
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels["input_ids"]  # Esto es crítico para el entrenamiento
        }
    
    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))
    
    # Mapeo en lotes pequeños
    train_ds = train_ds.map(
        preprocess_function,
        batched=True,
        batch_size=8,  # Tamaño de lote reducido
        remove_columns=["input_text", "target_text"]
    )
    test_ds = test_ds.map(
        preprocess_function,
        batched=True,
        batch_size=8,
        remove_columns=["input_text", "target_text"]
    )
    
    # 4. Configuración de entrenamiento optimizada
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    
    args = TrainingArguments(
        output_dir=f"./{nombre_modelo}_results",
        per_device_train_batch_size=2,  # Reducido de 4
        per_device_eval_batch_size=2,
        num_train_epochs=3,  # Reducido de 4
        learning_rate=2e-4,  # Tasa de aprendizaje más baja para estabilidad
        logging_dir=f"./{nombre_modelo}_logs",
        logging_steps=20,  # Menos frecuente
        save_total_limit=1,
        optim="adafactor",  # Optimizador eficiente para CPU
        report_to="none",  # Deshabilitar reporting para ahorrar recursos
        fp16=False,  # Deshabilitar para CPU
        gradient_accumulation_steps=2,  # Para compensar batch size pequeño
    )
    
    # 5. Entrenamiento con gestión de memoria
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        callbacks=[EarlyStoppingCallback(early_stopping_threshold)] if early_stopping_threshold else []
    )
    
    try:
        trainer.train()
    finally:
        # Guardar modelo y limpiar memoria
        model.save_pretrained(f"./{nombre_modelo}")
        tokenizer.save_pretrained(f"./{nombre_modelo}")
        del model, trainer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
    
    print(f"✅ Modelo guardado en ./{nombre_modelo}")

# =============================
# 6. ENTRENAMIENTO POR LOTES (OPTIMIZADO)
# =============================
if __name__ == "__main__":
    # Configuración optimizada para CPU
    config_modelos = [
        #{"etiqueta": "PRIORIDAD", "nombre_modelo": "modelo_t5_prioridad"},
        {"etiqueta": "IDX", "nombre_modelo": "modelo_t5_idx"},
        #{"etiqueta": "DERIVACION", "nombre_modelo": "modelo_t5_derivacion"},
        #{"etiqueta": "ESPECIALIDAD", "nombre_modelo": "modelo_t5_especialidad"}
    ]
    
    for config in config_modelos:
        print(f"\n{'='*40}\nIniciando entrenamiento para {config['etiqueta']}\n{'='*40}")
        entrenar_t5(
            df=df,
            etiqueta=config["etiqueta"],
            nombre_modelo=config["nombre_modelo"],
            balancear=True,
            early_stopping_threshold=0.03
        )
        
        # Limpieza entre modelos
        gc.collect()