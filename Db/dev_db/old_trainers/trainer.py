# =============================
# 1. IMPORTAR LIBRERÍAS
# =============================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, TrainerCallback
from datasets import Dataset
import tkinter as tk
from tkinter import filedialog

# =============================
# 2. DEFINIR CUSTOM CALLBACK PARA EARLY STOPPING
# =============================
class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, threshold):
        self.threshold = threshold
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs:
            if logs['loss'] < self.threshold:
                print(f"\n⚡ Loss alcanzó {logs['loss']:.4f} (umbral: {self.threshold}). Deteniendo entrenamiento...")
                control.should_training_stop = True

# =============================
# 3. SELECCIONAR ARCHIVO CSV
# =============================
print("🔼 Selecciona el archivo CSV con tus datos")
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(title="Selecciona el archivo CSV", filetypes=[("CSV Files", "*.csv")])
df = pd.read_csv(file_path)
print(f"✅ Archivo cargado: {file_path}")

# =============================
# 4. GENERAR TEXTO DE ENTRADA
# =============================
df = df.dropna(subset=["MOTIVO", "EDAD", "GENERO", "FC", "FR", "TAS", "SAO2", "TEMP", "NEWS2"])

df["input_text"] = (
    "Motivo: " + df["MOTIVO"].astype(str) + " | " +
    "Edad: " + df["EDAD"].astype(str) + " | " +
    "Genero: " + df["GENERO"].astype(str) + " | " +
    "FC: " + df["FC"].astype(str) + " | " +
    "FR: " + df["FR"].astype(str) + " | " +
    "TAS: " + df["TAS"].astype(str) + " | " +
    "SAO2: " + df["SAO2"].astype(str) + " | " +
    "TEMP: " + df["TEMP"].astype(str) + " | " +
    "NEWS2: " + df["NEWS2"].astype(str)
)

# =============================
# 5. FUNCIÓN GENERAL DE ENTRENAMIENTO MEJORADA
# =============================
def entrenar_t5(df, etiqueta, nombre_modelo, balancear=True, early_stopping_threshold=None):
    print(f"\n🚀 Entrenando T5 para: {etiqueta}")
    
    # Ver distribución de clases antes de procesar
    print("\nDistribución original de clases:")
    print(df[etiqueta].value_counts(normalize=True))
    
    df_temp = df.dropna(subset=[etiqueta]).copy()
    df_temp["target_text"] = df_temp[etiqueta].astype(str)
    df_task = df_temp[["input_text", "target_text"]]

    # Balanceo de clases (activado por defecto)
    if balancear:
        print("\n⚖️ Aplicando balanceo de clases...")
        
        # Separar por clases
        classes = df_task['target_text'].value_counts()
        max_size = classes.max()
        
        lst_dfs = []
        for class_label in classes.index:
            df_class = df_task[df_task['target_text'] == class_label]
            if len(df_class) < max_size:
                # Oversampling de clases minoritarias
                df_upsampled = resample(df_class, 
                                      replace=True, 
                                      n_samples=max_size, 
                                      random_state=42)
                lst_dfs.append(df_upsampled)
            else:
                lst_dfs.append(df_class)
        
        df_task = pd.concat(lst_dfs)
        print("\nDistribución después de balanceo:")
        print(df_task['target_text'].value_counts(normalize=True))

    # División estratificada para mantener proporciones en train/test
    train_df, test_df = train_test_split(df_task, test_size=0.1, random_state=42, stratify=df_task['target_text'])
    
    print("\nDistribución en conjunto de entrenamiento:")
    print(train_df['target_text'].value_counts(normalize=True))
    print("\nDistribución en conjunto de prueba:")
    print(test_df['target_text'].value_counts(normalize=True))
    
    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))

    tokenizer = AutoTokenizer.from_pretrained("t5-small")

    def preprocess(examples):
        inputs = examples["input_text"]
        targets = examples["target_text"]
        model_inputs = tokenizer(inputs, max_length=512, padding="max_length", truncation=True)
        labels = tokenizer(targets, max_length=32, padding="max_length", truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_ds = train_ds.map(preprocess, batched=True)
    test_ds = test_ds.map(preprocess, batched=True)

    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

    # Configurar callbacks (early stopping opcional para todos los modelos)
    callbacks = []
    if early_stopping_threshold is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_threshold))
        print(f"\n🔎 Early stopping activado - Se detendrá cuando loss < {early_stopping_threshold}")

    args = TrainingArguments(
        output_dir=f"./{nombre_modelo}_results",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=4,
        learning_rate=3e-4,
        logging_dir=f"./{nombre_modelo}_logs",
        logging_steps=10,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        callbacks=callbacks
    )

    trainer.train()

    model.save_pretrained(f"./{nombre_modelo}")
    tokenizer.save_pretrained(f"./{nombre_modelo}")
    print(f"✅ Modelo guardado en ./{nombre_modelo}")

# =============================
# 6. ENTRENAR TODOS LOS MODELOS CON CONFIGURACIÓN FLEXIBLE
# =============================
# Configuración para cada modelo (etiqueta, nombre_modelo, early_stopping_threshold)
config_modelos = [
    {"etiqueta": "PRIORIDAD", "nombre_modelo": "modelo_t5_prioridad", "early_stopping": 0.03},
    #{"etiqueta": "IDX", "nombre_modelo": "modelo_t5_idx", "early_stopping": None},
    #{"etiqueta": "DERIVACION", "nombre_modelo": "modelo_t5_derivacion", "early_stopping": 0.03},
    #{"etiqueta": "ESPECIALIDAD", "nombre_modelo": "modelo_t5_especialidad", "early_stopping": 0.03}
]

for config in config_modelos:
    entrenar_t5(
        df=df,
        etiqueta=config["etiqueta"],
        nombre_modelo=config["nombre_modelo"],
        balancear=True,  # Balanceo activado para todos
        early_stopping_threshold=config["early_stopping"]
    )