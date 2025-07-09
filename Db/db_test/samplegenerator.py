import pandas as pd
import tkinter as tk
from tkinter import filedialog
import os

def seleccionar_archivo():
    root = tk.Tk()
    root.withdraw()
    archivo = filedialog.askopenfilename(
        title="Selecciona el archivo CSV",
        filetypes=[("Archivos CSV", "*.csv")]
    )
    return archivo

def generar_muestra_estratificada(df):
    # Definir el tamaño de muestra por estrato (prioridad)
    tamanos_muestra = {
        'I': 20,
        'II': 30,
        'III': 75,
        'IV': 25
    }

    # Verificar que hay suficientes casos para cada estrato
    for prioridad, tamano in tamanos_muestra.items():
        disponibles = len(df[df['PRIORIDAD'] == prioridad])
        if disponibles < tamano:
            print(f"⚠️ Advertencia: Solo hay {disponibles} casos de prioridad {prioridad} (se solicitaban {tamano})")
            tamanos_muestra[prioridad] = disponibles  # Ajustar al máximo disponible

    # Seleccionar muestras aleatorias para cada estrato
    muestras = []
    for prioridad, tamano in tamanos_muestra.items():
        estrato = df[df['PRIORIDAD'] == prioridad]
        muestra = estrato.sample(n=tamano, random_state=42)  # random_state para reproducibilidad
        muestras.append(muestra)

    # Combinar todas las muestras
    muestra_final = pd.concat(muestras)

    # Mezclar el dataframe resultante
    return muestra_final.sample(frac=1, random_state=42).reset_index(drop=True)

def guardar_muestra(df, archivo_original):
    # Crear nombre para el archivo de muestra
    directorio = os.path.dirname(archivo_original)
    nombre_base = os.path.basename(archivo_original)
    nombre_muestra = "muestra_estratificada_" + nombre_base

    ruta_completa = os.path.join(directorio, nombre_muestra)

    # Guardar el nuevo CSV
    df.to_csv(ruta_completa, index=False, encoding='utf-8-sig')
    return ruta_completa

if __name__ == "__main__":
    print("📊 Generador de Muestra Estratificada por Prioridad")
    print("--------------------------------------------------")

    # Seleccionar archivo CSV
    csv_file = seleccionar_archivo()

    if csv_file:
        # Leer el archivo CSV
        try:
            df = pd.read_csv(csv_file, encoding='utf-8')

            # Verificar que existe la columna PRIORIDAD
            if 'PRIORIDAD' not in df.columns:
                raise ValueError("El archivo CSV no contiene la columna 'PRIORIDAD'")

            # Generar la muestra estratificada
            muestra = generar_muestra_estratificada(df)

            # Mostrar resumen
            print("\n📝 Resumen de la muestra generada:")
            print(muestra['PRIORIDAD'].value_counts().sort_index())

            # Guardar el nuevo archivo
            ruta_guardado = guardar_muestra(muestra, csv_file)
            print(f"\n✅ Muestra guardada en: {ruta_guardado}")

        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
    else:
        print("\n❌ No se seleccionó ningún archivo CSV")
