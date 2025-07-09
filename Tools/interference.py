import pandas as pd
import tkinter as tk
from tkinter import filedialog
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os
import time

# Configuración inicial con más detalles
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚙️ Dispositivo seleccionado: {device}")
print(f"⚙️ Versión de PyTorch: {torch.__version__}")
print(f"⚙️ Transformers disponible: {'transformers' in globals()}")

def seleccionar_archivo():
    try:
        root = tk.Tk()
        root.withdraw()
        archivo = filedialog.askopenfilename(
            title="Selecciona el archivo CSV con la muestra", 
            filetypes=[("CSV files", "*.csv")]
        )
        if archivo:
            print(f"📁 Archivo seleccionado: {archivo}")
            if not os.path.exists(archivo):
                print("❌ El archivo no existe en la ruta especificada")
                return None
            if os.path.getsize(archivo) == 0:
                print("❌ El archivo está vacío")
                return None
        return archivo
    except Exception as e:
        print(f"❌ Error al seleccionar archivo: {str(e)}")
        return None

def cargar_modelo(ruta_modelo):
    """Carga un modelo y su tokenizer desde la ruta especificada con validaciones"""
    print(f"\n🔍 Intentando cargar modelo desde: {ruta_modelo}")
    
    if not os.path.exists(ruta_modelo):
        raise Exception(f"La ruta del modelo no existe: {ruta_modelo}")
    
    try:
        print("⏳ Cargando tokenizer...")
        inicio = time.time()
        tokenizer = AutoTokenizer.from_pretrained(ruta_modelo)
        print(f"✅ Tokenizer cargado en {time.time() - inicio:.2f}s")
        
        print("⏳ Cargando modelo...")
        inicio = time.time()
        model = AutoModelForSeq2SeqLM.from_pretrained(ruta_modelo).to(device)
        print(f"✅ Modelo cargado en {time.time() - inicio:.2f}s")
        
        # Verificación rápida del modelo
        print("🧪 Realizando prueba rápida del modelo...")
        try:
            test_input = tokenizer("Prueba", return_tensors="pt").to(device)
            _ = model.generate(**test_input, max_length=5)
            print("✅ Prueba de modelo exitosa")
        except Exception as e:
            print(f"⚠️ Advertencia en prueba de modelo: {str(e)}")
        
        return tokenizer, model
    except Exception as e:
        raise Exception(f"Error al cargar el modelo en {ruta_modelo}: {str(e)}")

def verificar_dataframe(df):
    """Verifica que el DataFrame tenga las columnas necesarias"""
    print("\n🔍 Verificando estructura del DataFrame...")
    columnas_requeridas = {
        'MOTIVO': str,
        'EDAD': (int, float),
        'GENERO': str,
        'FC': (int, float),
        'FR': (int, float),
        'TAS': (int, float),
        'SAO2': (int, float),
        'TEMP': (int, float),
        'NEWS2': (int, float)
    }
    
    faltantes = [col for col in columnas_requeridas if col not in df.columns]
    if faltantes:
        raise ValueError(f"Columnas faltantes: {faltantes}")
    
    print("✅ Estructura básica del DataFrame es correcta")
    
    # Verificar muestras de datos
    print("\n📝 Muestra de datos (primer registro):")
    primer_registro = df.iloc[0]
    for col in columnas_requeridas:
        print(f"{col}: {primer_registro[col]} (tipo: {type(primer_registro[col])})")
    
    return True

def preparar_texto_entrada(fila):
    """Genera el texto de entrada para el modelo a partir de una fila del DataFrame"""
    try:
        texto = (
            f"Motivo: {fila['MOTIVO']} | "
            f"Edad: {fila['EDAD']} | "
            f"Genero: {fila['GENERO']} | "
            f"FC: {fila['FC']} | "
            f"FR: {fila['FR']} | "
            f"TAS: {fila['TAS']} | "
            f"SAO2: {fila['SAO2']} | "
            f"TEMP: {fila['TEMP']} | "
            f"NEWS2: {fila['NEWS2']}"
        )
        print("\n📄 Texto de entrada generado:")
        print(texto)
        return texto
    except Exception as e:
        raise ValueError(f"Error al preparar texto de entrada: {str(e)}")

def predecir_con_modelo(texto, tokenizer, model, max_length=32):
    """Realiza una predicción con el modelo especificado con logging detallado"""
    try:
        print("\n🔮 Comenzando predicción...")
        inicio = time.time()
        
        print("⏳ Tokenizando entrada...")
        inputs = tokenizer(texto, return_tensors="pt", max_length=512, truncation=True).to(device)
        print(f"✅ Tokenización completada en {time.time() - inicio:.2f}s")
        
        print("⏳ Generando predicción...")
        inicio_gen = time.time()
        outputs = model.generate(**inputs, max_length=max_length)
        print(f"✅ Predicción generada en {time.time() - inicio_gen:.2f}s")
        
        resultado = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"📌 Resultado de la predicción: '{resultado}'")
        
        return resultado
    except Exception as e:
        raise Exception(f"Error durante la predicción: {str(e)}")

def main():
    print("\n🔮 Script de Inferencia para Modelos de Triage - Versión Debug")
    print("="*70)
    
    # 1. Seleccionar archivo con la muestra
    print("\n🔄 PASO 1: Seleccionar archivo CSV")
    archivo_muestra = seleccionar_archivo()
    if not archivo_muestra:
        print("❌ No se seleccionó ningún archivo válido")
        return
    
    try:
        # 2. Cargar la muestra
        print("\n🔄 PASO 2: Cargar y validar datos")
        print(f"⏳ Cargando archivo {archivo_muestra}...")
        df = pd.read_csv(archivo_muestra, encoding='utf-8')
        print(f"✅ Datos cargados: {len(df)} registros, {len(df.columns)} columnas")
        
        verificar_dataframe(df)
        
        # 3. Cargar modelos entrenados
        print("\n🔄 PASO 3: Cargar modelos entrenados")
        modelos_a_cargar = {
            'PRIORIDAD': "./modelo_t5_prioridad",
            'DERIVACION': "./modelo_t5_derivacion",
            'ESPECIALIDAD': "./modelo_t5_especialidad",
            'IDX': "./modelo_t5_idx"
        }
        
        modelos = {}
        for nombre, ruta in modelos_a_cargar.items():
            print(f"\n🔍 Verificando modelo {nombre}...")
            if not os.path.exists(ruta):
                print(f"❌ No se encontró el modelo {nombre} en {ruta}")
                print("ℹ️ Contenido del directorio:")
                print(os.listdir(os.path.dirname(ruta)))
                continue
            
            try:
                modelos[nombre] = cargar_modelo(ruta)
                print(f"✅ Modelo {nombre} cargado correctamente")
            except Exception as e:
                print(f"❌ Error al cargar modelo {nombre}: {str(e)}")
                continue
        
        if not modelos:
            raise Exception("No se pudo cargar ningún modelo")
        
        # 4. Realizar predicciones para cada registro
        print("\n🔄 PASO 4: Realizar predicciones")
        resultados = []
        total_registros = len(df)
        
        for i, (_, fila) in enumerate(df.iterrows(), 1):
            print(f"\n📊 Procesando registro {i}/{total_registros}")
            try:
                # Preparar texto de entrada
                texto_entrada = preparar_texto_entrada(fila)
                
                # Realizar predicciones solo con modelos cargados correctamente
                preds = {}
                for nombre in modelos:
                    try:
                        print(f"\n🔮 Prediciendo {nombre}...")
                        inicio = time.time()
                        tokenizer, model = modelos[nombre]
                        preds[f'IA_{nombre}'] = predecir_con_modelo(texto_entrada, tokenizer, model)
                        print(f"✅ Predicción {nombre} completada en {time.time() - inicio:.2f}s")
                    except Exception as e:
                        print(f"❌ Error al predecir {nombre}: {str(e)}")
                        preds[f'IA_{nombre}'] = "ERROR"
                
                resultados.append(preds)
                
            except Exception as e:
                print(f"❌ Error procesando registro {i}: {str(e)}")
                resultados.append({f'IA_{k}': "ERROR" for k in modelos})
        
        # 5. Agregar predicciones al DataFrame original
        print("\n🔄 PASO 5: Consolidar resultados")
        df_resultados = pd.concat([df, pd.DataFrame(resultados)], axis=1)
        
        # 6. Guardar resultados
        print("\n🔄 PASO 6: Guardar resultados")
        nombre_base = os.path.basename(archivo_muestra)
        nombre_resultado = f"resultados_IA_{nombre_base}"
        ruta_guardado = os.path.join(os.path.dirname(archivo_muestra), nombre_resultado)
        
        print(f"⏳ Guardando resultados en {ruta_guardado}...")
        df_resultados.to_csv(ruta_guardado, index=False, encoding='utf-8-sig')
        print(f"✅ Resultados guardados exitosamente")
        print(f"📊 Resumen:\n{df_resultados.head()}")
        
    except Exception as e:
        print(f"\n❌❌❌ Error crítico durante el proceso: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inicio_total = time.time()
    main()
    print(f"\n🕒 Tiempo total de ejecución: {time.time() - inicio_total:.2f} segundos")