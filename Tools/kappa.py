import pandas as pd
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración visualización
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def cargar_datos(ruta_csv):
    """Carga y verifica la estructura del DataFrame"""
    df = pd.read_csv(ruta_csv, encoding='utf-8')
    
    # Verificar columnas necesarias
    columnas_requeridas = {
        'originales': ['PRIORIDAD', 'DERIVACION', 'ESPECIALIDAD', 'IDX'],
        'ia': ['IA_PRIORIDAD', 'IA_DERIVACION', 'IA_ESPECIALIDAD', 'IA_IDX']
    }
    
    for grupo in columnas_requeridas:
        faltantes = [col for col in columnas_requeridas[grupo] if col not in df.columns]
        if faltantes:
            raise ValueError(f"Columnas {grupo} faltantes: {faltantes}")
    
    return df

def calcular_kappa(df, col_original, col_ia):
    """Calcula y reporta el Kappa para un par de columnas"""
    print(f"\n🔍 Comparando: {col_original} vs {col_ia}")
    
    # Preparar datos
    y_true = df[col_original].astype(str).str.strip().str.upper()
    y_pred = df[col_ia].astype(str).str.strip().str.upper()
    
    # Calcular Kappa
    kappa = cohen_kappa_score(y_true, y_pred)
    
    # Interpretación cualitativa
    if kappa < 0:
        interpretacion = "Concordancia peor que aleatoria"
    elif 0 <= kappa <= 0.20:
        interpretacion = "Concordancia ligera"
    elif 0.21 <= kappa <= 0.40:
        interpretacion = "Concordancia moderada"
    elif 0.41 <= kappa <= 0.60:
        interpretacion = "Concordancia sustancial"
    else:
        interpretacion = "Concordancia casi perfecta"
    
    # Resultados
    print(f"📊 Kappa: {kappa:.4f}")
    print(f"📌 Interpretación: {interpretacion}")
    
    return kappa

def generar_visualizacion(resultados):
    """Genera gráficos de los resultados"""
    df_resultados = pd.DataFrame(resultados, columns=['Variable', 'Kappa'])
    
    # Gráfico de barras
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(x='Variable', y='Kappa', data=df_resultados, palette="Blues_d")
    plt.title('Concordancia entre Clasificaciones Humanas y de IA\n(Coeficiente Kappa de Cohen)')
    plt.ylim(0, 1)
    
    # Añadir valores en las barras
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}", 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                   textcoords='offset points')
    
    plt.savefig('resultados_kappa.png', dpi=300, bbox_inches='tight')
    print("\n📈 Gráfico guardado como: resultados_kappa.png")

def main():
    print("\n🔬 Análisis Estadístico - Coeficiente Kappa de Cohen")
    print("="*60)
    
    # 1. Cargar datos
    ruta_csv = input("Ingrese la ruta del archivo CSV con los resultados: ").strip('"')
    try:
        df = cargar_datos(ruta_csv)
        print(f"\n✅ Datos cargados correctamente. Registros: {len(df)}")
        print("Columnas disponibles:", df.columns.tolist())
        
        # 2. Calcular Kappa para cada par de variables
        resultados = []
        variables = ['PRIORIDAD', 'DERIVACION', 'ESPECIALIDAD', 'IDX']
        
        for var in variables:
            col_original = var
            col_ia = f"IA_{var}"
            
            kappa = calcular_kappa(df, col_original, col_ia)
            resultados.append({'Variable': var, 'Kappa': kappa})
        
        # 3. Generar reporte visual
        generar_visualizacion(resultados)
        
        # 4. Exportar resultados numéricos
        df_kappa = pd.DataFrame(resultados)
        df_kappa.to_csv('kappa_results.csv', index=False)
        print("\n📝 Resultados numéricos guardados como: kappa_results.csv")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    main()