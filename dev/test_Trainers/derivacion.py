import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, get_scheduler
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import logging
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración básica
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DualOutputT5WithTabular(nn.Module):
    def __init__(self, t5_model, n_tab_features, n_deriv_classes, n_espec_classes):
        super().__init__()
        self.t5 = t5_model
        hidden_size = self.t5.config.d_model

        # Capa para características tabulares
        self.tabular_layer = nn.Sequential(
            nn.Linear(n_tab_features, 64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Fusión para derivación
        self.deriv_fusion = nn.Linear(hidden_size + 64, hidden_size)
        self.deriv_classifier = nn.Linear(hidden_size, n_deriv_classes)
        
        # Fusión para especialidad
        self.espec_fusion = nn.Linear(hidden_size + 64, hidden_size)
        self.espec_classifier = nn.Linear(hidden_size, n_espec_classes)

    def forward(self, input_ids, attention_mask, features_num):
        # Encoder T5
        outputs = self.t5.encoder(input_ids=input_ids, attention_mask=attention_mask)
        encoder_hidden_states = outputs.last_hidden_state

        # Procesar características tabulares
        tab_out = self.tabular_layer(features_num).unsqueeze(1)
        batch_size, seq_len, hid_dim = encoder_hidden_states.size()
        
        # Fusionar para derivación
        deriv_cat = torch.cat([encoder_hidden_states, tab_out.expand(-1, seq_len, -1)], dim=2)
        deriv_features = self.deriv_fusion(deriv_cat)
        deriv_logits = self.deriv_classifier(deriv_features[:, 0, :])  # Usamos solo el primer token
        
        # Fusionar para especialidad
        espec_cat = torch.cat([encoder_hidden_states, tab_out.expand(-1, seq_len, -1)], dim=2)
        espec_features = self.espec_fusion(espec_cat)
        espec_logits = self.espec_classifier(espec_features[:, 0, :])  # Usamos solo el primer token
        
        return deriv_logits, espec_logits

def train_dual_model():
    """Función para entrenar el modelo unificado"""
    try:
        logger.info("\n=== ENTRENANDO MODELO UNIFICADO PARA DERIVACIÓN Y ESPECIALIDAD ===")
        
        # --- 1. Cargar y preprocesar datos
        df = pd.read_csv("db.csv")
        df = df.dropna(subset=["MOTIVO", "DERIVACION", "ESPECIALIDAD", "EDAD", "GENERO", "TAS", "TAD", "FC", "FR", "TEMP", "SAO2", "NEWS2"])
        
        # --- 2. Convertir variables categóricas a numéricas ---
        # Mapear género (igual que en modelos anteriores)
        def map_genero(g):
            if isinstance(g, str):
                g = g.upper()
                if g == "HOMBRE":
                    return 1
                elif g == "MUJER":
                    return 0
            return 0  # Desconocido o no especificado = 0
        
        df["GENERO"] = df["GENERO"].apply(map_genero)
        # -----------------------------------------------------
        
        # Limpieza de datos numéricos
        numeric_cols = ["EDAD", "TAS", "TAD", "FC", "FR", "TEMP", "SAO2", "NEWS2"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=numeric_cols)
        
        # Codificar variables objetivo
        le_deriv = LabelEncoder()
        df["DERIVACION_LABEL"] = le_deriv.fit_transform(df["DERIVACION"])
        
        le_espec = LabelEncoder()
        df["ESPECIALIDAD_LABEL"] = le_espec.fit_transform(df["ESPECIALIDAD"])
        
        # Verificar distribución de clases
        logger.info("\nDistribución de clases para DERIVACION:")
        logger.info(df["DERIVACION_LABEL"].value_counts())
        
        logger.info("\nDistribución de clases para ESPECIALIDAD:")
        logger.info(df["ESPECIALIDAD_LABEL"].value_counts())
        
        # --- 3. Escalar características numéricas ---
        # Asegurarse que GENERO ya es numérico (0, 1)
        features_num_cols = ["EDAD", "GENERO", "TAS", "TAD", "FC", "FR", "TEMP", "SAO2", "NEWS2"]
        
        # Verificar tipos de datos antes de escalar
        logger.info("\nTipos de datos antes de escalar:")
        logger.info(df[features_num_cols].dtypes)
        
        scaler = StandardScaler()
        df[features_num_cols] = scaler.fit_transform(df[features_num_cols])
        # -------------------------------------------
        
        # Dividir datos
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, 
                                          stratify=df[["DERIVACION_LABEL", "ESPECIALIDAD_LABEL"]])
        
        # --- 2. Configuración del modelo
        model_name = "google/mt5-small"
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        base_model = T5ForConditionalGeneration.from_pretrained(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Crear modelo dual
        model = DualOutputT5WithTabular(
            base_model,
            n_tab_features=len(features_num_cols),
            n_deriv_classes=len(le_deriv.classes_),
            n_espec_classes=len(le_espec.classes_)
        ).to(device)
        
        # --- 3. Dataset y DataLoader
        class CustomDataset(Dataset):
            def __init__(self, dataframe, tokenizer, max_len=128):
                self.df = dataframe
                self.tokenizer = tokenizer
                self.max_len = max_len
                self.features_num_cols = features_num_cols

            def __len__(self): return len(self.df)
            
            def __getitem__(self, idx):
                row = self.df.iloc[idx]
                text = str(row["MOTIVO"])
                
                inputs = self.tokenizer(text, max_length=self.max_len, padding="max_length", 
                                      truncation=True, return_tensors="pt")
                
                features_num = torch.tensor(row[self.features_num_cols].values.astype(np.float32), 
                                dtype=torch.float)
                
                return {
                    "input_ids": inputs["input_ids"].squeeze(0),
                    "attention_mask": inputs["attention_mask"].squeeze(0),
                    "features_num": features_num,
                    "deriv_label": row["DERIVACION_LABEL"],
                    "espec_label": row["ESPECIALIDAD_LABEL"]
                }

        train_dataset = CustomDataset(train_df, tokenizer)
        val_dataset = CustomDataset(val_df, tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8)
        
        # --- 4. Configuración de entrenamiento
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        num_epochs = 4
        num_training_steps = num_epochs * len(train_loader)
        lr_scheduler = get_scheduler("linear", optimizer=optimizer, 
                                    num_warmup_steps=0, num_training_steps=num_training_steps)
        
        # Funciones de pérdida
        deriv_weights = torch.tensor(
            compute_class_weight('balanced', 
                               classes=np.unique(df["DERIVACION_LABEL"]),
                               y=df["DERIVACION_LABEL"]),
            dtype=torch.float
        ).to(device)
        
        espec_weights = torch.tensor(
            compute_class_weight('balanced', 
                               classes=np.unique(df["ESPECIALIDAD_LABEL"]),
                               y=df["ESPECIALIDAD_LABEL"]),
            dtype=torch.float
        ).to(device)
        
        deriv_loss_fn = nn.CrossEntropyLoss(weight=deriv_weights)
        espec_loss_fn = nn.CrossEntropyLoss(weight=espec_weights)
        
        # --- 5. Entrenamiento
        os.makedirs("saved_models/dual_model", exist_ok=True)
        best_val_accuracy = 0.0
        train_losses = []
        val_metrics = []
        
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                optimizer.zero_grad()
                
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                features_num = batch["features_num"].to(device)
                deriv_labels = batch["deriv_label"].to(device)
                espec_labels = batch["espec_label"].to(device)
                
                deriv_logits, espec_logits = model(input_ids, attention_mask, features_num)
                
                loss = deriv_loss_fn(deriv_logits, deriv_labels) + espec_loss_fn(espec_logits, espec_labels)
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                epoch_loss += loss.item()
            
            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validación
            model.eval()
            deriv_correct = 0
            espec_correct = 0
            total = 0
            val_loss = 0.0
            all_deriv_preds = []
            all_espec_preds = []
            all_deriv_true = []
            all_espec_true = []
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validando"):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    features_num = batch["features_num"].to(device)
                    deriv_labels = batch["deriv_label"].to(device)
                    espec_labels = batch["espec_label"].to(device)
                    
                    deriv_logits, espec_logits = model(input_ids, attention_mask, features_num)
                    
                    loss = deriv_loss_fn(deriv_logits, deriv_labels) + espec_loss_fn(espec_logits, espec_labels)
                    val_loss += loss.item()
                    
                    deriv_preds = torch.argmax(deriv_logits, dim=1)
                    espec_preds = torch.argmax(espec_logits, dim=1)
                    
                    deriv_correct += (deriv_preds == deriv_labels).sum().item()
                    espec_correct += (espec_preds == espec_labels).sum().item()
                    total += len(deriv_labels)
                    
                    all_deriv_preds.extend(deriv_preds.cpu().numpy())
                    all_espec_preds.extend(espec_preds.cpu().numpy())
                    all_deriv_true.extend(deriv_labels.cpu().numpy())
                    all_espec_true.extend(espec_labels.cpu().numpy())
            
            avg_val_loss = val_loss / len(val_loader)
            deriv_acc = deriv_correct / total
            espec_acc = espec_correct / total
            avg_acc = (deriv_acc + espec_acc) / 2
            
            val_metrics.append({
                'epoch': epoch+1,
                'val_loss': avg_val_loss,
                'deriv_acc': deriv_acc,
                'espec_acc': espec_acc,
                'avg_acc': avg_acc
            })
            
            logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
            logger.info(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            logger.info(f"Derivación Accuracy: {deriv_acc:.4f}")
            logger.info(f"Especialidad Accuracy: {espec_acc:.4f}")
            logger.info(f"Accuracy Promedio: {avg_acc:.4f}")
            
            # Guardar mejor modelo
            if avg_acc > best_val_accuracy:
                best_val_accuracy = avg_acc
                torch.save(model.state_dict(), "saved_models/dual_model/best_model.pth")
                joblib.dump(le_deriv, "saved_models/dual_model/deriv_encoder.pkl")
                joblib.dump(le_espec, "saved_models/dual_model/espec_encoder.pkl")
                joblib.dump(scaler, "saved_models/dual_model/scaler.pkl")
                
                # Guardar reportes de clasificación
                deriv_report = classification_report(all_deriv_true, all_deriv_preds, 
                                                   target_names=le_deriv.classes_, output_dict=True)
                espec_report = classification_report(all_espec_true, all_espec_preds, 
                                                   target_names=le_espec.classes_, output_dict=True)
                
                pd.DataFrame(deriv_report).transpose().to_csv("saved_models/dual_model/deriv_report.csv")
                pd.DataFrame(espec_report).transpose().to_csv("saved_models/dual_model/espec_report.csv")
                
                # Guardar matrices de confusión
                plt.figure(figsize=(12, 5))
                plt.subplot(1, 2, 1)
                sns.heatmap(confusion_matrix(all_deriv_true, all_deriv_preds), 
                           annot=True, fmt='d', cmap='Blues',
                           xticklabels=le_deriv.classes_, 
                           yticklabels=le_deriv.classes_)
                plt.title('Matriz de Confusión - Derivación')
                plt.xticks(rotation=45)
                
                plt.subplot(1, 2, 2)
                sns.heatmap(confusion_matrix(all_espec_true, all_espec_preds), 
                           annot=True, fmt='d', cmap='Blues',
                           xticklabels=le_espec.classes_, 
                           yticklabels=le_espec.classes_)
                plt.title('Matriz de Confusión - Especialidad')
                plt.xticks(rotation=45)
                
                plt.tight_layout()
                plt.savefig("saved_models/dual_model/confusion_matrices.png")
                plt.close()
        
        # --- 6. Resultados finales y gráficas
        logger.info("\n=== RESULTADOS FINALES ===")
        logger.info(f"Mejor Accuracy Promedio: {best_val_accuracy:.4f}")
        
        # Gráfica de métricas
        plt.figure(figsize=(15, 5))
        
        # Pérdidas
        plt.subplot(1, 3, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot([m['val_loss'] for m in val_metrics], label='Val Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Accuracy por tarea
        plt.subplot(1, 3, 2)
        plt.plot([m['deriv_acc'] for m in val_metrics], label='Derivación')
        plt.plot([m['espec_acc'] for m in val_metrics], label='Especialidad')
        plt.plot([m['avg_acc'] for m in val_metrics], label='Promedio')
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Gráfica de barras final
        plt.subplot(1, 3, 3)
        final_metrics = val_metrics[-1]
        acc_values = [final_metrics['deriv_acc'], final_metrics['espec_acc'], final_metrics['avg_acc']]
        acc_labels = ['Derivación', 'Especialidad', 'Promedio']
        plt.bar(acc_labels, acc_values)
        plt.title('Final Validation Accuracy')
        plt.ylabel('Accuracy')
        
        plt.tight_layout()
        plt.savefig("saved_models/dual_model/training_metrics.png")
        plt.close()
        
        logger.info("\nModelo unificado entrenado y guardado en 'saved_models/dual_model'")
        
    except Exception as e:
        logger.error(f"Error entrenando modelo unificado: {e}", exc_info=True)

# Ejecutar entrenamiento
train_dual_model()