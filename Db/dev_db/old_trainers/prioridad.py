import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, get_scheduler
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import joblib
from huggingface_hub import notebook_login
import logging
from tqdm import tqdm
import os

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # --- 1. Autenticación en Hugging Face (si es necesario)
    notebook_login()

    # --- 2. Cargar y preprocesar base
    logger.info("Cargando y preprocesando datos...")
    df = pd.read_csv("db.csv")
    
    # Verificar y limpiar datos
    logger.info("Verificando y limpiando datos...")
    required_cols = ["MOTIVO", "PRIORIDAD", "CDX", "EDAD", "GENERO", "TAS", "TAD", "FC", "FR", "TEMP", "SAO2", "NEWS2"]
    df = df.dropna(subset=required_cols)
    
    # Convertir columnas numéricas y manejar valores no numéricos
    numeric_cols = ["EDAD", "TAS", "TAD", "FC", "FR", "TEMP", "SAO2", "NEWS2"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Eliminar filas con valores nulos después de la conversión
    df = df.dropna(subset=numeric_cols)
    
    # Mapear género
    def map_genero(g):
        if isinstance(g, str):
            g = g.upper()
            if g == "HOMBRE":
                return 1
            elif g == "MUJER":
                return 0
        return 0  # Desconocido o no especificado = 0

    df["GENERO"] = df["GENERO"].apply(map_genero)

    # Codificar prioridad
    le_prioridad = LabelEncoder()
    df["PRIORIDAD_LABEL"] = le_prioridad.fit_transform(df["PRIORIDAD"])
    
    # Verificar distribución de clases
    logger.info(f"Distribución de clases: {df['PRIORIDAD_LABEL'].value_counts()}")

    # Escalar características numéricas
    features_num_cols = ["EDAD", "GENERO", "TAS", "TAD", "FC", "FR", "TEMP", "SAO2", "NEWS2"]
    scaler = StandardScaler()
    df[features_num_cols] = scaler.fit_transform(df[features_num_cols])

    # Dividir datos
    train_df, val_df = train_test_split(df, stratify=df["PRIORIDAD_LABEL"], test_size=0.2, random_state=42)
    logger.info(f"Datos de entrenamiento: {len(train_df)}, Datos de validación: {len(val_df)}")

    # --- 3. Tokenizador y modelo T5
    model_name = "google/mt5-small"  # Modelo oficial de Google
    
    try:
        logger.info(f"Cargando tokenizer y modelo {model_name}...")
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
    except Exception as e:
        logger.error(f"Error al cargar el modelo: {e}")
        # Alternativa de respaldo
        logger.info("Intentando con otro modelo...")
        model_name = "t5-small"
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Modelo cargado en dispositivo: {device}")

    # --- 4. Dataset personalizado
    class CustomDataset(Dataset):
        def __init__(self, dataframe, tokenizer, max_len=128):
            self.df = dataframe.reset_index(drop=True)
            self.tokenizer = tokenizer
            self.max_len = max_len
            self.features_num_cols = features_num_cols

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.loc[idx]
            text = str(row["MOTIVO"])  # Asegurar que es string

            inputs = self.tokenizer(
                text, 
                max_length=self.max_len, 
                padding="max_length", 
                truncation=True, 
                return_tensors="pt"
            )
            
            # Asegurar que el texto objetivo sea string
            target_text = str(row["PRIORIDAD"])
            targets = self.tokenizer(
                target_text, 
                max_length=8, 
                padding="max_length", 
                truncation=True, 
                return_tensors="pt"
            )

            # Convertir características numéricas asegurando que son float32
            features_num = torch.tensor(
                row[self.features_num_cols].values.astype(np.float32), 
                dtype=torch.float
            )

            item = {
                "input_ids": inputs["input_ids"].squeeze(0),
                "attention_mask": inputs["attention_mask"].squeeze(0),
                "labels": targets["input_ids"].squeeze(0),
                "features_num": features_num
            }

            return item

    train_dataset = CustomDataset(train_df, tokenizer)
    val_dataset = CustomDataset(val_df, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)  # Reducido batch_size para evitar problemas de memoria
    val_loader = DataLoader(val_dataset, batch_size=8)

    # --- 5. Modelo extendido para incluir tabulares
    class T5WithTabular(nn.Module):
        def __init__(self, t5_model, n_tab_features):
            super().__init__()
            self.t5 = t5_model
            hidden_size = self.t5.config.d_model

            self.tabular_layer = nn.Sequential(
                nn.Linear(n_tab_features, 64),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            self.decoder_fusion = nn.Linear(hidden_size + 64, hidden_size)

        def forward(self, input_ids, attention_mask, labels, features_num):
            outputs = self.t5.encoder(input_ids=input_ids, attention_mask=attention_mask)
            encoder_hidden_states = outputs.last_hidden_state

            tab_out = self.tabular_layer(features_num).unsqueeze(1)
            batch_size, seq_len, hid_dim = encoder_hidden_states.size()
            enc_cat = torch.cat([encoder_hidden_states, tab_out.expand(-1, seq_len, -1)], dim=2)
            fused_encoder_states = self.decoder_fusion(enc_cat)

            decoder_outputs = self.t5.decoder(
                input_ids=labels[:, :-1],
                encoder_hidden_states=fused_encoder_states,
                encoder_attention_mask=attention_mask,
            )
            sequence_output = decoder_outputs.last_hidden_state
            logits = self.t5.lm_head(sequence_output)

            loss = None
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss(ignore_index=self.t5.config.pad_token_id)
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels[:, 1:].reshape(-1))

            return loss, logits

    model_extended = T5WithTabular(model, n_tab_features=len(features_num_cols)).to(device)

    # --- 6. Pesos balanceados para la pérdida
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(df["PRIORIDAD_LABEL"]),
        y=df["PRIORIDAD_LABEL"]
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    class WeightedCrossEntropyLoss(nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.weight = weight
            self.loss_fct = nn.CrossEntropyLoss(weight=weight, ignore_index=tokenizer.pad_token_id)

        def forward(self, logits, labels):
            return self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

    loss_fn = WeightedCrossEntropyLoss(weight=class_weights)

    # --- 7. Optimizer y scheduler
    optimizer = torch.optim.AdamW(model_extended.parameters(), lr=2e-5)
    num_epochs = 4
    num_training_steps = num_epochs * len(train_loader)

    lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # --- 8. Entrenamiento
    logger.info("Iniciando entrenamiento...")
    
    # Crear directorio para guardar modelos
    os.makedirs("saved_models", exist_ok=True)
    
    best_val_accuracy = 0.0
    train_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model_extended.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            features_num = batch["features_num"].to(device)

            loss, logits = model_extended(input_ids, attention_mask, labels, features_num)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")

        # Validación
        model_extended.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validando"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                features_num = batch["features_num"].to(device)

                loss, logits = model_extended(input_ids, attention_mask, labels, features_num)
                val_loss += loss.item()
                
                preds = torch.argmax(logits, dim=-1)
                correct += (preds[:, 0] == labels[:, 1]).sum().item()
                total += labels.size(0)

        val_accuracy = correct / total
        val_accuracies.append(val_accuracy)
        avg_val_loss = val_loss / len(val_loader)
        
        logger.info(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

        # Guardar el mejor modelo
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model_extended.state_dict(), f"saved_models/best_model_epoch_{epoch+1}.pth")
            logger.info(f"Nuevo mejor modelo guardado con accuracy: {val_accuracy:.4f}")

    # --- 9. Guardar modelo final y artefactos
    logger.info("Guardando modelo final y artefactos...")
    torch.save(model_extended.state_dict(), "saved_models/final_model.pth")
    joblib.dump(le_prioridad, "saved_models/encoder_prioridad.pkl")
    joblib.dump(scaler, "saved_models/scaler_num.pkl")

    # Guardar métricas
    metrics = {
        "train_losses": train_losses,
        "val_accuracies": val_accuracies,
        "best_val_accuracy": best_val_accuracy
    }
    joblib.dump(metrics, "saved_models/training_metrics.pkl")

    # Mostrar resumen de entrenamiento
    logger.info("\n=== Resumen del Entrenamiento ===")
    logger.info(f"Mejor precisión en validación: {best_val_accuracy:.4f}")
    logger.info(f"Pérdida final en entrenamiento: {train_losses[-1]:.4f}")
    logger.info(f"Precisión final en validación: {val_accuracies[-1]:.4f}")
    
    # Graficar métricas (opcional, requiere matplotlib)
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(val_accuracies, label='Validation Accuracy')
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        
        plt.tight_layout()
        plt.savefig("saved_models/training_metrics.png")
        plt.close()
        logger.info("Gráficas de métricas guardadas en saved_models/training_metrics.png")
    except ImportError:
        logger.warning("Matplotlib no está instalado. No se generarán gráficas.")

    logger.info("Entrenamiento completo y artefactos guardados en la carpeta 'saved_models'")

except Exception as e:
    logger.error(f"Error en el proceso principal: {e}", exc_info=True)