import gradio as gr
import torch
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    T5Tokenizer, T5ForConditionalGeneration
)

# =============================
# CONFIGURACIÓN DE GOOGLE SHEETS
# =============================
SHEET_NAME = "TriageIA"
SCOPES = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
CREDS_FILE = "service_account.json"

def conectar_hoja():
    creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, SCOPES)
    cliente = gspread.authorize(creds)
    hoja = cliente.open(SHEET_NAME).sheet1
    return hoja

# =============================
# CARGA DE MODELOS
# =============================
modelo_clas = AutoModelForSequenceClassification.from_pretrained("drpablo-hospital/modelo_cb_prioridad")
tokenizer_clas = AutoTokenizer.from_pretrained("drpablo-hospital/modelo_cb_prioridad")

modelo_idx = T5ForConditionalGeneration.from_pretrained("drpablo-hospital/modelo_t5_idx")
tok_idx = T5Tokenizer.from_pretrained("drpablo-hospital/modelo_t5_idx")

modelo_esp = AutoModelForSequenceClassification.from_pretrained("drpablo-hospital/modelo_cb_especialidad")
tok_esp = AutoTokenizer.from_pretrained("drpablo-hospital/modelo_cb_especialidad")

modelo_der = AutoModelForSequenceClassification.from_pretrained("drpablo-hospital/modelo_cb_derivacion")
tok_der = AutoTokenizer.from_pretrained("drpablo-hospital/modelo_cb_derivacion")

id2prioridad = {0: "I", 1: "II", 2: "III", 3: "IV"}
id2derivacion = {0: "ALTA", 1: "CONSULTA DE", 2: "INGRESA A"}
id2especialidad = {0: "CIRUGIA", 1: "OFTALMOLOGIA", 2: "ORTOPEDIA", 3: "OTORRINOLARINGOLOGIA", 4: "RECONSTRUCTIVA", 5: "SIN ESPECIALIDAD", 6: "URGENCIAS", 7: "UROLOGIA"}

# =============================
# FUNCIONES AUXILIARES
# =============================
def agrupar_diagnostico(texto):
    texto = str(texto).upper()
    if 'ABDOMINAL' in texto or 'ABDOMEN' in texto:
        return 'DOLOR ABDOMINAL'
    elif 'FRACTURA' in texto:
        return 'FRACTURA'
    elif 'INTOXICACION' in texto:
        return 'INTOXICACION'
    elif 'HIPERTENSO' in texto or 'HAS' in texto:
        return 'CRISIS HIPERTENSIVA'
    elif 'DIABETES' in texto or 'GLICEMIA' in texto:
        return 'ALTERACION GLUCEMICA'
    elif 'SANGRADO' in texto:
        return 'SANGRADO'
    elif 'CAIDA' in texto:
        return 'CAIDA / TRAUMA'
    elif 'PARO' in texto or 'PCR' in texto:
        return 'PARO CARDIORRESPIRATORIO'
    elif 'DOLOR TORACICO' in texto or 'PRECORDIAL' in texto:
        return 'DOLOR TORACICO'
    elif 'EPILEPSIA' in texto or 'CONVULSION' in texto:
        return 'CONVULSIONES'
    elif 'DISNEA' in texto or 'DIFICULTAD RESPIRATORIA' in texto:
        return 'DISNEA'
    else:
        return 'OTROS'

def generar_pistas(motivo, ecg, tas, fc, sao2, news2):
    pistas = []
    if ecg < 13:
        pistas.append("🧠Posible emergencia neurológica.🧠")
    if tas < 90:
        pistas.append("💡 Podría requerir atención urgente. 💡")
    if fc > 140:
        pistas.append("🫀 Posible taquicardia severa. Ekg inmediato. 🫀")
    if sao2 < 90:
        pistas.append("🪫 Riesgo de hipoxemia crítica. 🪫")
    if news2 > 8:
        pistas.append("🚨 Prioridad vital: Ingreso urgente. 🚨")
    elif news2 > 6:
        pistas.append("🚧 Requiere evaluación inmediata. 🚧")
    elif news2 > 4:
        pistas.append("⌛ Caso con alerta fisiológica moderada. ⌛")
    motivo_l = motivo.lower()
    if 'diabet' in motivo_l and ('descontrol' in motivo_l or 'hiperglucemia' in motivo_l):
        pistas.append("🧪 Emergencia metabólica probable. 🧪")
    if 'toracico' in motivo_l or 'precordial' in motivo_l:
        pistas.append("🫀 Riesgo de infarto. Ekg inmediato. 🫀")
    if 'hipoglucemia' in motivo_l and ('alerta' in motivo_l or 'respuesta' in motivo_l):
        pistas.append ("🧠 Posible emergencia neurológica. 🧠")
    return pistas

def generar_folio():
    hoja = conectar_hoja()
    hoy = datetime.now().strftime("%d%m%y")
    filas = hoja.get_all_values()
    conteo_hoy = sum(1 for fila in filas if fila and fila[0].startswith(hoy))
    return f"{hoy}{conteo_hoy+1:03d}"

# =============================
# FUNCIÓN PRINCIPAL DE PREDICCIÓN
# =============================
def predecir(edad, genero, motivo, ecg, tas, tad, fc, fr, temp, sao2):
    news2 = 0
    if ecg <= 8: news2 += 3
    elif ecg <= 10: news2 += 2
    elif ecg <= 12: news2 += 1
    if fr <= 8 or fr >= 25: news2 += 3
    elif fr >= 21: news2 += 2
    if sao2 < 92: news2 += 3
    elif sao2 < 94: news2 += 2
    elif sao2 < 96: news2 += 1
    if tas <= 90: news2 += 3
    elif tas <= 100: news2 += 2
    elif tas <= 110: news2 += 1
    if temp < 35 or temp > 39.1: news2 += 2
    elif temp < 36 or temp > 38: news2 += 1
    if fc <= 40 or fc > 130: news2 += 3
    elif fc > 110: news2 += 2
    elif fc > 90 or fc < 50: news2 += 1

    pistas = generar_pistas(motivo, ecg, tas, fc, sao2, news2)
    pistas_txt = " ".join(pistas)

    texto = (
        f"Paciente de {edad} años, género {genero}. "
        f"Motivo de consulta: {motivo}. "
        f"Signos vitales: ECG {ecg}, TA {tas}/{tad}, "
        f"FC {fc}, FR {fr}, TEMP {temp}°C, SpO2 {sao2}%. "
    )

    # PRIORIDAD
    inputs_clas = tokenizer_clas(texto, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    with torch.no_grad():
        logits = modelo_clas(**inputs_clas).logits
        prioridad = id2prioridad[torch.argmax(logits, dim=1).item()]

    # DERIVACIÓN
    inputs_der = tok_der(texto, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    with torch.no_grad():
        logits = modelo_der(**inputs_der).logits
        derivacion = id2derivacion[torch.argmax(logits, dim=1).item()]

    # ESPECIALIDAD
    inputs_esp = tok_esp(texto, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    with torch.no_grad():
        logits = modelo_esp(**inputs_esp).logits
        especialidad = id2especialidad[torch.argmax(logits, dim=1).item()]

    # DIAGNÓSTICO T5
    idx = tok_idx.decode(
        modelo_idx.generate(tok_idx(texto, return_tensors="pt").input_ids, max_length=32)[0],
        skip_special_tokens=True
    )

    # GUARDADO
    folio = generar_folio()
    import pytz
    hora_ingreso = datetime.now(pytz.timezone("America/Mexico_City")).strftime("%Y-%m-%d %H:%M:%S")

    # IMPRIMIR EN TERMINAL PARA VERIFICACIÓN
    print("\n===== DATOS A GUARDAR =====")
    print(f"Folio: {folio}")
    print(f"Edad: {edad}")
    print(f"Género: {genero}")
    print(f"Motivo de consulta: {motivo}")
    print(f"ECG: {ecg}")
    print(f"TAS: {tas}")
    print(f"TAD: {tad}")
    print(f"FC: {fc}")
    print(f"FR: {fr}")
    print(f"Temperatura: {temp}")
    print(f"SpO2: {sao2}")
    print(f"NEWS2: {news2}")
    print(f"Prioridad: {prioridad}")
    print(f"Derivación: {derivacion}")
    print(f"Especialidad: {especialidad}")
    print(f"IDX: {idx}")
    print(f"Hora de ingreso: {hora_ingreso}")
    print("============================\n")

    hoja = conectar_hoja()
    hoja.append_row([
        folio, edad, genero, motivo, ecg, tas, tad, fc, fr, temp, sao2, news2,
        prioridad, derivacion, especialidad, idx, hora_ingreso, ""
    ])


    return f"FOLIO: {folio}", prioridad, derivacion, especialidad, idx, pistas_txt if pistas else "—"

# =============================
# INTERFAZ
# =============================
demo = gr.Interface(
    fn=predecir,
    inputs=[
        gr.Number(label="Edad"),
        gr.Radio(["MUJER", "HOMBRE"], label="Género"),
        gr.Textbox(label="Motivo de consulta"),
        gr.Number(label="ECG"),
        gr.Number(label="TAS (sistólica)"),
        gr.Number(label="TAD (diastólica)"),
        gr.Number(label="FC"),
        gr.Number(label="FR"),
        gr.Number(label="Temperatura (°C)"),
        gr.Number(label="SpO2 (%)")
    ],
    outputs=[
        gr.Textbox(label="FOLIO"),
        gr.Textbox(label="PRIORIDAD"),
        gr.Textbox(label="DERIVACIÓN"),
        gr.Textbox(label="ESPECIALIDAD"),
        gr.Textbox(label="DIAGNÓSTICO (IDX)"),
        gr.Textbox(label="PISTAS CLÍNICAS")
    ],
    title="Triage Enfermería 🪔",
    description="Captura de pacientes y predicción clínica automatizada con IA"
)

if __name__ == "__main__":
    demo.launch()
