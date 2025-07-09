# =============================
# EVALUACIÓN COMPLETA DEL MODELO
# =============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from datasets import Dataset
import torch

# 1. CARGAR MODELO Y DATOS
# ========================
model_path = "./modelo_prioridad"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Cargar datos de prueba (asegúrate de tener una columna "texto_clinico" y "PRIORIDAD_real")
df_test = pd.read_csv("dbmod.csv")  # Ajusta la ruta

# 2. PREDICCIONES
# ===============
def predecir(textos):
    inputs = tokenizer(textos, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.softmax(outputs.logits, dim=1).numpy()

probabilidades = predecir(df_test["texto_clinico"].tolist())
predicciones = np.argmax(probabilidades, axis=1)

# Mapeo inverso de índices a etiquetas (ajusta según tu modelo)
id2label = {0: "Prioridad_1", 1: "Prioridad_2", 2: "Prioridad_3"}  # Ejemplo

# 3. MÉTRICAS ESTADÍSTICAS
# ========================
y_true = df_test["PRIORIDAD_real"].map(lambda x: list(prioridad2id.keys())[list(prioridad2id.values()).index(x)])
y_pred = [id2label[p] for p in predicciones]

print("\n📊 REPORTE DE CLASIFICACIÓN:")
print(classification_report(y_true, y_pred, target_names=id2label.values()))

# 4. MATRIZ DE CONFUSIÓN (HEATMAP)
# ================================
plt.figure(figsize=(10, 7))
matriz_conf = confusion_matrix(y_true, y_pred)
sns.heatmap(matriz_conf, 
            annot=True, 
            fmt="d", 
            cmap="Blues",
            xticklabels=id2label.values(), 
            yticklabels=id2label.values())
plt.title("Matriz de Confusión")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.savefig("matriz_confusion.png")  # Guardar imagen
plt.show()

# 5. CURVA DE APRENDIZAJE (USANDO HISTORIAL DE PÉRDIDA)
# =====================================================
# Extraer loss del entrenamiento (simulado - ajusta con tus datos reales)
loss_history = [0.696, 0.55, 0.35, 0.27]  # Ejemplo: loss por epoch

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(loss_history)+1), loss_history, marker='o')
plt.title("Curva de Pérdida durante el Entrenamiento")
plt.xlabel("Época")
plt.ylabel("Loss")
plt.grid()
plt.savefig("curva_perdida.png")
plt.show()

# 6. DISTRIBUCIÓN DE PROBABILIDADES
# =================================
plt.figure(figsize=(10, 5))
for i, label in enumerate(id2label.values()):
    sns.kdeplot(probabilidades[:, i], label=label)
plt.title("Distribución de Probabilidades por Clase")
plt.xlabel("Probabilidad")
plt.ylabel("Densidad")
plt.legend()
plt.savefig("distribucion_probabilidades.png")
plt.show()

# 7. TOP ERRORES
# ==============
df_test["prediccion"] = y_pred
df_test["correcto"] = (y_true == y_pred)

# Mostrar 5 casos con mayor error
top_errores = df_test[df_test["correcto"] == False].sort_values(
    by=[np.max(probabilidades, axis=1)], 
    ascending=False
).head(5)

print("\n🔴 TOP 5 ERRORES:")
print(top_errores[["texto_clinico", "PRIORIDAD_real", "prediccion"]])