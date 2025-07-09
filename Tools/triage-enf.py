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
# Clasificación de prioridad - BERT
modelo_clas = AutoModelForSequenceClassification.from_pretrained("drpablo-hospital/modelo_cb_prioridad")
tokenizer_clas = AutoTokenizer.from_pretrained("drpablo-hospital/modelo_cb_prioridad")

# Diagnóstico IDX - T5
modelo_idx = T5ForConditionalGeneration.from_pretrained("drpablo-hospital/modelo_t5_idx")
tok_idx = T5Tokenizer.from_pretrained("drpablo-hospital/modelo_t5_idx")

# Especialidad - BERT
modelo_esp = AutoModelForSequenceClassification.from_pretrained("drpablo-hospital/modelo_cb_especialidad")
tok_esp = AutoTokenizer.from_pretrained("drpablo-hospital/modelo_cb_especialidad")

# Derivación - BERT
modelo_der = AutoModelForSequenceClassification.from_pretrained("drpablo-hospital/modelo_cb_derivacion")
tok_der = AutoTokenizer.from_pretrained("drpablo-hospital/modelo_cb_derivacion")

id2prioridad = {0: "I", 1: "II", 2: "III", 3: "IV", 4: "V"}

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

    idx_grupo = agrupar_diagnostico(motivo)
    pistas = generar_pistas(motivo, ecg, tas, fc, sao2, news2)
    pistas_txt = " ".join(pistas)

    texto = (
        f"Paciente de {edad} años, género {genero}. "
        f"Motivo de consulta: {motivo}. "
        f"Signos vitales: ECG {ecg}, TA {tas}/{tad}, "
        f"FC {fc}, FR {fr}, TEMP {temp}°C, SpO2 {sao2}%. "
        f"Diagnóstico estimado: {idx_grupo}. {pistas_txt}"
    )

    # PRIORIDAD (BERT)
    inputs_clas = tokenizer_clas(texto, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    with torch.no_grad():
        logits_clas = modelo_clas(**inputs_clas).logits
        pred_id = torch.argmax(logits_clas, dim=1).item()
        prioridad = id2prioridad[pred_id]

    # DERIVACIÓN (BERT)
    inputs_der = tok_der(texto, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    with torch.no_grad():
        logits_der = modelo_der(**inputs_der).logits
        der = tok_der.decode(torch.argmax(logits_der, dim=1), skip_special_tokens=True)

    # ESPECIALIDAD (BERT)
    inputs_esp = tok_esp(texto, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    with torch.no_grad():
        logits_esp = modelo_esp(**inputs_esp).logits
        esp = tok_esp.decode(torch.argmax(logits_esp, dim=1), skip_special_tokens=True)

    # DIAGNÓSTICO IDX (T5)
    inputs_idx = tok_idx(texto, return_tensors="pt")
    with torch.no_grad():
        idx_ids = modelo_idx.generate(inputs_idx.input_ids, max_length=32)
        idx = tok_idx.decode(idx_ids[0], skip_special_tokens=True)

    # REGISTRO
    folio = generar_folio()
    import pytz
    hora_ingreso = datetime.now(pytz.timezone("America/Mexico_City")).strftime("%Y-%m-%d %H:%M:%S")
    hoja = conectar_hoja()
    hoja.append_row([
        folio, edad, genero, motivo, ecg, tas, tad, fc, fr, temp, sao2, news2,
        prioridad, der, esp, idx, hora_ingreso, ""
    ])

    return f"FOLIO: {folio}", prioridad, der, esp, idx, pistas_txt if pistas else "—"

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
