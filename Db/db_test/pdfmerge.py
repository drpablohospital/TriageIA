import os
from PyPDF2 import PdfMerger
import tkinter as tk
from tkinter import filedialog

def seleccionar_carpeta():
    root = tk.Tk()
    root.withdraw()
    carpeta = filedialog.askdirectory(title="Selecciona la carpeta con los PDFs")
    return carpeta

def unificar_pdfs(carpeta_origen, archivo_salida="pdf_unificado.pdf"):
    # Crear un objeto PdfMerger
    merger = PdfMerger()

    # Listar todos los archivos PDF en la carpeta
    pdfs = [archivo for archivo in os.listdir(carpeta_origen) if archivo.lower().endswith('.pdf')]

    if not pdfs:
        print("⚠️ No se encontraron archivos PDF en la carpeta seleccionada")
        return

    print(f"🔍 Encontrados {len(pdfs)} archivos PDF para unificar...")

    # Procesar cada PDF
    for pdf in pdfs:
        try:
            ruta_completa = os.path.join(carpeta_origen, pdf)
            merger.append(ruta_completa)
            print(f"✓ Añadido: {pdf}")
        except Exception as e:
            print(f"✗ Error al procesar {pdf}: {str(e)}")

    # Guardar el PDF unificado
    ruta_salida = os.path.join(carpeta_origen, archivo_salida)
    merger.write(ruta_salida)
    merger.close()

    print(f"\n✅ PDFs unificados correctamente en: {ruta_salida}")
    print(f"📄 Total de páginas: {len(merger.pages)}")

if __name__ == "__main__":
    print("🔄 Iniciando proceso de unificación de PDFs")
    carpeta = seleccionar_carpeta()
    if carpeta:
        unificar_pdfs(carpeta)
    else:
        print("❌ No se seleccionó ninguna carpeta")
