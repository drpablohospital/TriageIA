import os
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import tkinter as tk
from tkinter import filedialog

def seleccionar_archivo():
    root = tk.Tk()
    root.withdraw()
    archivo = filedialog.askopenfilename(
        title="Selecciona el archivo PDF",
        filetypes=[("Archivos PDF", "*.pdf")]
    )
    return archivo

def pdf_a_texto(pdf_path, txt_path=None):
    """
    Convierte un archivo PDF a texto usando OCR (recomendado para PDF escaneados)
    o extracción directa de texto (para PDF con texto editable).
    """
    if txt_path is None:
        txt_path = os.path.splitext(pdf_path)[0] + ".txt"

    # Configuración de Tesseract (ajusta la ruta según tu instalación)
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    try:
        # Primero intentamos extracción directa de texto (para PDF editables)
        import PyPDF2
        with open(pdf_path, 'rb') as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n\n"

            # Si extrajo texto suficiente, lo guardamos
            if len(text.strip()) > 100:  # Si tiene más de 100 caracteres
                with open(txt_path, 'w', encoding='utf-8') as txt_file:
                    txt_file.write(text)
                print(f"✅ Texto extraído directamente y guardado en: {txt_path}")
                return

        # Si no se extrajo suficiente texto, usamos OCR
        print("🔍 El PDF parece ser escaneado, aplicando OCR...")
        images = convert_from_path(pdf_path)
        full_text = ""

        for i, image in enumerate(images, start=1):
            print(f"Procesando página {i}/{len(images)}...")
            text = pytesseract.image_to_string(image, lang='spa')  # 'spa' para español
            full_text += f"--- Página {i} ---\n{text}\n\n"

        with open(txt_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(full_text)

        print(f"✅ Texto extraído con OCR y guardado en: {txt_path}")

    except Exception as e:
        print(f"❌ Error durante la conversión: {str(e)}")

if __name__ == "__main__":
    print("📄 Conversor de PDF a TXT")
    print("-------------------------")

    # Seleccionar archivo PDF
    pdf_file = seleccionar_archivo()

    if pdf_file:
        # Crear nombre para el archivo TXT
        txt_file = os.path.splitext(pdf_file)[0] + ".txt"

        # Convertir PDF a TXT
        pdf_a_texto(pdf_file, txt_file)

        print("\nProceso completado. Puedes abrir el archivo resultante:")
        print(txt_file)
    else:
        print("❌ No se seleccionó ningún archivo PDF")
