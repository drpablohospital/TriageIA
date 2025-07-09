import pandas as pd
import unicodedata

# Carga tu archivo CSV (cambia la ruta)
archivo = "dbtest-unclean.csv"
df = pd.read_csv(archivo, encoding='utf-8', dtype=str)  # dtype=str para tratar todo como texto

# Función para limpiar caracteres raros
def limpiar_texto(texto):
    if pd.isnull(texto):
        return texto  # deja los NaN intactos
    # Normaliza el texto, elimina acentos y convierte a ASCII simple
    texto_limpio = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('utf-8')
    return texto_limpio

# Aplica la función a todo el DataFrame
df = df.applymap(limpiar_texto)

# Guarda el resultado
df.to_csv("archivo_limpio.csv", index=False, encoding='utf-8')

print("¡Listo! Archivo limpio guardado como 'archivo_limpio.csv'")
