import pandas as pd

def listar_idx_unicos(archivo_csv):
    # Leer el archivo CSV
    try:
        df = pd.read_csv(archivo_csv, delimiter='\t' if '\t' in open(archivo_csv).read() else ',')
    except Exception as e:
        print(f"Error al leer el archivo: {e}")
        return
    
    # Verificar si existe la columna IDX
    if 'IDX' not in df.columns:
        print("Error: No se encontró la columna 'IDX' en el archivo.")
        return
    
    # Obtener valores únicos de IDX, ordenados alfabéticamente
    idx_unicos = df['IDX'].str.strip().str.upper().unique()
    idx_unicos_ordenados = sorted(idx_unicos)
    
    # Mostrar resultados
    print(f"\nSe encontraron {len(idx_unicos_ordenados)} valores únicos en la columna IDX:\n")
    for i, idx in enumerate(idx_unicos_ordenados, 1):
        print(f"{i}. {idx}")
    
    # Guardar resultados en un archivo de texto
    with open('lista_idx_unicos.txt', 'w', encoding='utf-8') as f:
        f.write("Valores únicos en columna IDX:\n\n")
        for idx in idx_unicos_ordenados:
            f.write(f"- {idx}\n")
    
    print("\nSe ha guardado la lista completa en 'lista_idx_unicos.txt'")

# Uso del script
archivo = input("Ingresa la ruta de tu archivo CSV: ")
listar_idx_unicos(archivo)