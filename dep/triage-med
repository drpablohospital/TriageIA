import gradio as gr
import gspread
import json
from datetime import datetime
from oauth2client.service_account import ServiceAccountCredentials
import pytz
import tempfile
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph
from reportlab.lib.utils import ImageReader
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
import textwrap
import tempfile
import os
import re

SHEET_NAME = "TriageIA"
SHEET_BAJA = "TriageBaja"
SCOPES = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
CREDS_FILE = "service_account.json"

ENTIDADES = [
    "Aguascalientes", "Baja California", "Baja California Sur", "Campeche", "Chiapas", "Chihuahua",
    "Ciudad de México", "Coahuila", "Colima", "Durango", "Estado de México", "Guanajuato", "Guerrero",
    "Hidalgo", "Jalisco", "Michoacán", "Morelos", "Nayarit", "Nuevo León", "Oaxaca", "Puebla",
    "Querétaro", "Quintana Roo", "San Luis Potosí", "Sinaloa", "Sonora", "Tabasco", "Tamaulipas",
    "Tlaxcala", "Veracruz", "Yucatán", "Zacatecas", "Extranjero"
]

DESTINOS = [
    "Alta", "Centro de Salud", "Pre-Consulta", "Ingreso a Observación", "Ingreso a Choque",
    "Cirugía General", "Urología", "Ortopedia", "Cirugía Plástica y Reconstructiva",
    "Otorrinolaringología", "Oftalmología", "Fuga", "Referencia", "Contrarreferencia"
]

def conectar_hoja(nombre=SHEET_NAME):
    creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, SCOPES)
    cliente = gspread.authorize(creds)
    return cliente.open(SHEET_NAME).worksheet(nombre)

def listar_folios():
    hoja = conectar_hoja("TriageIA")
    filas = hoja.get_all_values()[1:]
    datos = []

    for fila in filas:
        try:
            folio = fila[0]
            edad = fila[1]
            genero = fila[2]
            motivo = fila[3]
            news2 = int(fila[11])
            prioridad = fila[12]
            hora = datetime.strptime(fila[16], "%Y-%m-%d %H:%M:%S")
            hora = pytz.timezone("America/Mexico_City").localize(hora)
            espera = int((datetime.now(pytz.timezone("America/Mexico_City")) - hora).total_seconds() // 60)

            prioridad_valor = {"I": 5, "II": 4, "III": 3, "IV": 2, "V": 1}.get(prioridad, 0)
            color_emoji = "🔴" if news2 >= 8 else "🟠" if news2 >= 6 else "🟡" if news2 >= 4 else "🟢" if news2 >= 1 else "🔵"

            texto = f"{color_emoji} {folio} | {edad}a | {genero} | NEWS2 {news2} | PRIORIDAD {prioridad} | ⏱️ {espera} min → {motivo}"


            datos.append((news2, prioridad_valor, espera, texto))
        except Exception as e:
            print(f"❌ Error al procesar fila: {e}")
            continue

    if not datos:
        return "❌ Sin registros disponibles"

    datos.sort(reverse=True)
    if not datos:
        return "❌ Sin registros disponibles"

    datos.sort(reverse=True)
    return gr.update(choices=[item[-1] for item in datos], value=None)

def buscar_folio(folio):
    hoja = conectar_hoja("TriageIA")
    data = hoja.get_all_values()
    for idx, fila in enumerate(data[1:], start=2):
        if fila[0] == folio:
            return fila, idx
    return None, None

def cargar_datos_paciente(seleccion):
    if not seleccion:
        return ["" for _ in range(14)]

    folio = re.search(r"\b\d{9}\b", seleccion).group(0)
    fila, _ = buscar_folio(folio)

    if not fila:
        print("⚠️ Fila vacía o folio no encontrado.")
        return ["Paciente no encontrado"] * 14

    # Forzar longitud de fila a mínimo 19 columnas
    while len(fila) < 19:
        fila.append("")

    # Construcción del resumen
    try:
        hora_ingreso = datetime.strptime(fila[16], "%Y-%m-%d %H:%M:%S")
        minutos = int((datetime.now() - hora_ingreso).total_seconds() // 60)
    except:
        minutos = 0

    resumen = f"""
🆔 FOLIO: {fila[0]}
Edad: {fila[1]} años | Género: {fila[2]}
Motivo: {fila[3]}
NEWS-2: {fila[11]}
Signos vitales:
- ECG: {fila[4]}
- TA: {fila[5]}/{fila[6]}
- FC: {fila[7]}
- FR: {fila[8]}
- TEMP: {fila[9]}
- SpO2: {fila[10]}
🟥 Prioridad: {fila[12]}
🧭 Derivación: {fila[13]}
🏥 Especialidad: {fila[14]}
📝 Diagnóstico: {fila[15]}
⏱ Tiempo de espera: {minutos} min
""".strip()

    # Intento de carga de nota
    try:
        raw = fila[17].strip()
        print(f"🟡 Capturado en fila[18]: {raw}")
        if raw.startswith("{") and raw.endswith("}"):
            nota = json.loads(raw)
        else:
            print(f"⚠️ Contenido no válido en nota_medica_json: {raw}")
            nota = {}
    except Exception as e:
        print(f"❌ Error al parsear JSON en nota_medica_json: {e}")
        nota = {}

    return [
        resumen,
        nota.get("nombres", ""),
        nota.get("apellido_paterno", ""),
        nota.get("apellido_materno", ""),
        nota.get("fecha_nacimiento", ""),
        nota.get("entidad", ENTIDADES[0]),
        nota.get("padecimiento", ""),
        nota.get("exploracion", ""),
        nota.get("plan", ""),
        nota.get("laboratorios", ""),
        nota.get("impresion_diagnostica", ""),
        nota.get("envio_a", DESTINOS[0]),
        nota.get("medico", ""),
        nota.get("cedula", "")
    ]
def guardar_nota_estructurada(folio_str, nombres, ape_pat, ape_mat, nacimiento, entidad, padecimiento, exploracion, plan, labs, impresion, destino, medico, cedula):
    folio = re.search(r"\b\d{9}\b", folio_str).group(0)
    _, idx = buscar_folio(folio)
    if not idx:
        return "❌ Folio no encontrado"

    nota = {
        "nombres": nombres,
        "apellido_paterno": ape_pat,
        "apellido_materno": ape_mat,
        "fecha_nacimiento": nacimiento,
        "entidad": entidad,
        "padecimiento": padecimiento,
        "exploracion": exploracion,
        "plan": plan,
        "laboratorios": labs,
        "impresion_diagnostica": impresion,
        "envio_a": destino,
        "medico": medico,
        "cedula": cedula,
        "ultima_actualizacion": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    hoja = conectar_hoja("TriageIA")
    hoja.update_cell(idx, 18, json.dumps(nota, ensure_ascii=False))
    return "✅ Nota médica guardada."

def generar_curp(nombre, ape_pat, ape_mat, nacimiento, genero, estado):
    estados = {
        "aguascalientes": "AS", "baja california": "BC", "baja california sur": "BS", "campeche": "CC",
        "chiapas": "CS", "chihuahua": "CH", "ciudad de méxico": "DF", "coahuila": "CL", "colima": "CM",
        "durango": "DG", "estado de méxico": "EM", "guanajuato": "GT", "guerrero": "GR", "hidalgo": "HG",
        "jalisco": "JC", "michoacán": "MC", "morelos": "MS", "nayarit": "NT", "nuevo león": "NL",
        "oaxaca": "OC", "puebla": "PL", "querétaro": "QT", "quintana roo": "QR", "san luis potosí": "SP",
        "sinaloa": "SL", "sonora": "SR", "tabasco": "TC", "tamaulipas": "TS", "tlaxcala": "TL",
        "veracruz": "VZ", "yucatán": "YN", "zacatecas": "ZS", "extranjero": "NE"
    }
    estado_clave = estados.get(estado.lower(), "NE")
    nombre = nombre.strip().upper()
    ape_pat = ape_pat.strip().upper()
    ape_mat = ape_mat.strip().upper()
    genero = genero.strip().upper()[0]
    nac = parse_fecha(nacimiento).strftime("%y%m%d")
    curp = (
        ape_pat[0] +
        next((c for c in ape_pat[1:] if c in "AEIOU"), "X") +
        (ape_mat[0] if ape_mat else "X") +
        (nombre[0] if nombre else "X") +
        nac + genero + estado_clave + "XX"
    )
    return curp.upper()

def parse_fecha(fecha_str):
    for fmt in ("%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(fecha_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Formato de fecha no reconocido: {fecha_str}")

def imprimir_documento(resumen, nombres, ape_pat, ape_mat, nacimiento, entidad,
                       padecimiento, exploracion, plan, labs, impresion,
                       destino, medico, cedula):
    genero = "MUJER" if destino.lower().startswith("g") else "HOMBRE"
    curp = generar_curp(nombres, ape_pat, ape_mat, nacimiento, genero, entidad)
    output_path = os.path.join(tempfile.gettempdir(), f"nota_urgencias_{curp}.pdf")
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    x_margin = 40
    y = height - 40
    line_height = 10
    triage_line_height = 12

    def draw_line(text, bold=False, center=False, size=10, x=None):
        nonlocal y
        if y < 60:
            c.showPage()
            y = height - 40
        font = "Helvetica-Bold" if bold else "Helvetica"
        c.setFont(font, size)
        draw_x = width / 2 if center else (x if x is not None else x_margin)
        if center:
            c.drawCentredString(draw_x, y, text)
        else:
            c.drawString(draw_x, y, text)
        y -= line_height

    def draw_wrapped(text, max_width, bold=False, x=None, size=8):
        nonlocal y
        font = "Helvetica-Bold" if bold else "Helvetica"
        c.setFont(font, size)
        wrap_width = int(max_width / c.stringWidth("A", font, size))  # estimación segura
        lines = textwrap.wrap(text, width=wrap_width)
        for line in lines:
            draw_line(line, bold=bold, x=x, size=size)

    # Logo
    logo_path = "gea-logo.png"
    logo_width = 80
    logo_height = 60
    if os.path.exists(logo_path):
        c.drawImage(ImageReader(logo_path), x_margin, y - logo_height, width=logo_width, height=logo_height, preserveAspectRatio=True, mask='auto')
    encabezado_y = y - 10

    # Encabezado
    y = encabezado_y
    draw_line("COMISIÓN COORDINADORA DE INSTITUTOS NACIONALES DE SALUD Y HOSPITALES DE ALTA ESPECIALIDAD", center=True, size=6)
    draw_line("DIRECCIÓN GENERAL DE COORDINACIÓN DE HOSPITALES FEDERALES DE REFERENCIA", center=True, size=6)
    draw_line("HOSPITAL GENERAL \"DR MANUEL GEA GONZALEZ\"", center=True, size=8)
    draw_line("CALZADA DE TLALPAN 4800, CIUDAD DE MÉXICO CLUES: DFSSA003961 | TEL: 55 4000 3000", center=True, size=6)
    draw_line(" ", center=True, size=8)
    draw_line("HOJA DE VALORACIÓN INICIAL EN URGENCIAS", bold=True, center=True, size=12)
    y -= line_height * 2

    # Identificación
    fecha_nac = parse_fecha(nacimiento)
    edad = int((datetime.now() - fecha_nac).days / 365.25)
    genero = "MUJER" if destino.lower().startswith("g") else "HOMBRE"
    curp = generar_curp(nombres, ape_pat, ape_mat, nacimiento, genero, entidad)

    draw_line("📌 IDENTIFICACIÓN:", bold=True)
    draw_line(f"Nombre completo: {nombres} {ape_pat} {ape_mat}")
    draw_line(f"Fecha de nacimiento: {nacimiento}")
    draw_line(f"Entidad de origen: {entidad}")
    draw_line(f"CURP: {curp}")
    y -= line_height * 2

    # TRIAGE con columnas
    draw_line("🩺 TRIAGE:", bold=True)
    left_x = x_margin
    right_x = width / 2 - 100
    triage_y_start = y

    c.setFont("Helvetica-Bold", 10)
    c.drawString(right_x, triage_y_start, f"{genero} DE {edad} AÑOS DE EDAD")
    c.setFont("Helvetica", 10)
    c.drawString(right_x, triage_y_start - line_height, f"Motivo de consulta: {resumen.split('Motivo: ')[-1].splitlines()[0].strip()}")
    c.setFont("Helvetica-Bold", 10)
    c.drawString(right_x, triage_y_start - 2 * line_height, f"NEWS-2 Score: {resumen.split('NEWS-2: ')[-1].splitlines()[0].strip()}")
    c.drawString(right_x, triage_y_start - 3 * line_height, f"Prioridad asignada: {resumen.split('Prioridad: ')[-1].splitlines()[0].strip()}")
    c.setFont("Helvetica-Bold", 10)
    c.drawString(left_x, triage_y_start, "Signos vitales:")
    c.setFont("Helvetica", 10)
    y_offset = triage_y_start
    for line in resumen.splitlines():
        if line.startswith("- "):
            y_offset -= triage_line_height
            valor = line.replace("- ", "")
            if "ECG" in valor:
                valor += " puntos"
            elif "TA" in valor:
                valor += " mmHg"
            elif "FC" in valor:
                valor += " lpm"
            elif "FR" in valor:
                valor += " cpm"
            elif "TEMP" in valor:
                valor += " °C"
            elif "SpO2" in valor:
                valor += " %"
            c.drawString(left_x, y_offset, valor)
            y = y_offset - triage_line_height * 2
            y -= line_height * 2

    # Nota médica
    draw_line("📝 NOTA MÉDICA:", bold=True)
    draw_line(f"Fecha y hora de nota: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    draw_wrapped(f"Padecimiento actual: {padecimiento}", max_width = width - 2 * x_margin - 10, size=10)
    if exploracion.strip():
        draw_wrapped(f"Exploración física: {exploracion}", max_width = width - 2 * x_margin - 10, size=9)
    if plan.strip():
        draw_wrapped(f"Plan y análisis: {plan}", max_width = width - 2 * x_margin - 10, size=9)
    if labs.strip():
        draw_wrapped(f"Laboratorios y gabinete: {labs}", max_width = width - 2 * x_margin - 10, size=9)
    if impresion.strip():
        draw_wrapped(f"Impresión diagnóstica: {impresion}", max_width = width - 2 * x_margin - 10, bold=True, size=10)
    y -= line_height

    draw_line("📤 DESTINO:", bold=True)
    draw_line(f"Se envía a: {destino}")
    y -= line_height * 2

    # Atención médica con médico adscrito alineado derecha
    draw_line("📤 ATENCIÓN MÉDICA:", bold=True)

    # Coordenadas
    left_x = x_margin
    right_x = width / 2 + 20  # puedes ajustar más o menos según necesites

    # Línea 1
    c.setFont("Helvetica", 8)
    c.drawString(left_x, y, f"Médico que brindó atención: {medico}")
    c.drawString(right_x, y, "Médico adscrito:")
    y -= line_height

    # Línea 2
    c.drawString(left_x, y, f"Cédula profesional: {cedula}")
    c.drawString(right_x, y, "Cédula profesional: ")
    y -= line_height

    # Línea 3
    c.drawString(left_x, y, "Firma del médico: ")
    c.drawString(right_x, y, "Firma del médico: ")
    y -= line_height * 10

    draw_line("📤 Acudir a urgencias en caso de datos de alarma 📤", center=True, size=6)
    draw_line("Recibí atención médica. He sido informado sobre mi estado de salud y entiendo las indicaciones médicas.", center=True, size=6)
    draw_line("Nombre del paciente o familiar responsable y firma: ", center=True, size=6)

    c.save()
    return gr.File(value=output_path, label="📄 Descargar Nota Médica")

def dar_de_baja(folio_str):
    folio = re.search(r"\b\d{9}\b", folio_str).group(0)
    hoja = conectar_hoja("TriageIA")
    hoja_baja = conectar_hoja("TriageBaja")
    data = hoja.get_all_values()

    for idx, fila in enumerate(data[1:], start=2):
        if fila[0] == folio:
            fila[0] = f"{folio}B"  # Asignar nuevo folio con sufijo B
            hoja_baja.append_row(fila)
            hoja.delete_rows(idx)
            return f"✅ Paciente con folio {folio} dado de baja como {folio}B"

    return "❌ Paciente no encontrado"

# INTERFAZ
with gr.Blocks() as demo:
    gr.Markdown("# 📝 VALORACIÓN INICIAL EN URGENCIAS")
    lista = gr.Dropdown(label="Pacientes activos", choices=[], interactive=True)
    btn_refrescar = gr.Button("🔄 Refrescar listado")
    resumen = gr.Textbox(label="Triage del paciente (Asistido por IA)", lines=12)

    gr.Markdown("## 👤 SECCIÓN 1: Datos de identidad")
    with gr.Row():
        nombres = gr.Textbox(label="Nombre(s)")
        ape_pat = gr.Textbox(label="Apellido paterno")
        ape_mat = gr.Textbox(label="Apellido materno")
    nacimiento = gr.Textbox(label="Fecha de nacimiento (YYYY-MM-DD)")
    entidad = gr.Dropdown(choices=ENTIDADES, label="Entidad de origen")

    gr.Markdown("## 📋 SECCIÓN 2: Datos clínicos")
    padecimiento = gr.Textbox(label="Padecimiento actual", lines=2)
    exploracion = gr.Textbox(label="Exploración física", lines=2)
    plan = gr.Textbox(label="Plan y análisis", lines=2)
    labs = gr.Textbox(label="Laboratorios y gabinete", lines=2)
    impresion = gr.Textbox(label="Impresión diagnóstica", lines=2)

    gr.Markdown("## 🩺 SECCIÓN 3: Resultado de valoración")
    destino = gr.Dropdown(choices=DESTINOS, label="Envío a")
    medico = gr.Textbox(label="Médico responsable")
    cedula = gr.Textbox(label="Cédula profesional")

    btn_guardar = gr.Button("💾 Guardar nota médica")
    btn_imprimir = gr.Button("🖨 Imprimir")
    btn_baja = gr.Button("🗑 Dar de baja")

    vista_pdf = gr.File()
    mensaje = gr.Textbox(label="Mensaje del sistema")

    btn_refrescar.click(fn=listar_folios, outputs=lista)
    lista.change(fn=cargar_datos_paciente, inputs=lista, outputs=[
        resumen, nombres, ape_pat, ape_mat, nacimiento, entidad,
        padecimiento, exploracion, plan, labs, impresion, destino, medico, cedula
    ])
    btn_guardar.click(fn=guardar_nota_estructurada, inputs=[
        lista, nombres, ape_pat, ape_mat, nacimiento, entidad,
        padecimiento, exploracion, plan, labs, impresion, destino, medico, cedula
    ], outputs=mensaje)
    btn_baja.click(fn=dar_de_baja, inputs=lista, outputs=mensaje)
    btn_imprimir.click(
    fn=imprimir_documento,
    inputs=[resumen, nombres, ape_pat, ape_mat, nacimiento, entidad,
            padecimiento, exploracion, plan, labs, impresion, destino, medico, cedula],
    outputs=vista_pdf
    )
    demo.load(fn=listar_folios, outputs=lista)

if __name__ == "__main__":
    demo.launch()
