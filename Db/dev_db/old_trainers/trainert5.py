# =============================
# 1. IMPORTAR LIBRERÍAS
# =============================
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import Dataset
import tkinter as tk
from tkinter import filedialog

# =============================
# 2. SELECCIONAR ARCHIVO CSV
# =============================
print("🔼 Selecciona el archivo CSV con tus datos")
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(title="Selecciona el archivo CSV", filetypes=[("CSV Files", "*.csv")])
df = pd.read_csv(file_path)
print(f"✅ Archivo cargado: {file_path}")

# =============================
# 3. GENERAR TEXTO DE ENTRADA
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
# 4. FUNCIÓN GENERAL DE ENTRENAMIENTO
# =============================
def entrenar_t5(df, etiqueta, nombre_modelo):
    print(f"\n🚀 Entrenando T5 para: {etiqueta}")
    
    df_temp = df.dropna(subset=[etiqueta]).copy()
    df_temp["target_text"] = df_temp[etiqueta].astype(str)
    df_task = df_temp[["input_text", "target_text"]]

    train_df, test_df = train_test_split(df_task, test_size=0.1, random_state=42)
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
        tokenizer=tokenizer
    )

    trainer.train()

    model.save_pretrained(f"./{nombre_modelo}")
    tokenizer.save_pretrained(f"./{nombre_modelo}")
    print(f"✅ Modelo guardado en ./{nombre_modelo}")

# =============================
# 5. ENTRENAR PARA CADA TAREA
# =============================
#entrenar_t5(df, "IDX", "modelo_t5_idx")
#entrenar_t5(df, "PRIORIDAD", "modelo_t5_prioridad")
#entrenar_t5(df, "DERIVACION", "modelo_t5_derivacion")
#entrenar_t5(df, "ESPECIALIDAD", "modelo_t5_especialidad")
