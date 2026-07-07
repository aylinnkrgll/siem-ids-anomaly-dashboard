import os
from io import BytesIO
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

st.set_page_config(page_title="IDS — Ağ Anomali Tespit Sistemi", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
.info-box{background:#F5F7FA;border-left:5px solid #1E88E5;padding:12px 16px;border-radius:6px;margin:8px 0}
.warning-box{background:#FFF8E1;border-left:5px solid #FFB300;padding:12px 16px;border-radius:6px;margin:8px 0}
.report-box{background:#F8FAFC;border:1px solid #E5E7EB;border-left:5px solid #64748B;padding:14px 18px;border-radius:8px;margin-top:10px}
.risk-high{background:#FFEBEE;border-left:6px solid #E53935;padding:14px 18px;border-radius:8px;margin:8px 0}
.risk-medium{background:#FFF8E1;border-left:6px solid #FFB300;padding:14px 18px;border-radius:8px;margin:8px 0}
.risk-low{background:#E8F5E9;border-left:6px solid #43A047;padding:14px 18px;border-radius:8px;margin:8px 0}
.model-card{background:#fff;border:1px solid #E5E7EB;border-radius:10px;padding:14px 16px;box-shadow:0 1px 2px rgba(0,0,0,.04);margin-bottom:8px}
.model-card h4{margin:0 0 6px 0}.model-card p{margin:3px 0;font-size:14px}.small-note{color:#6B7280;font-size:13px}
.hero-card{background:#fff;border:1px solid #E5E7EB;border-radius:12px;padding:14px 16px;box-shadow:0 1px 3px rgba(0,0,0,.06);min-height:88px}
.hero-card .label{font-size:13px;color:#6B7280;font-weight:600}.hero-card .value{font-size:26px;font-weight:800;margin-top:4px}.hero-card .hint{font-size:12px;color:#6B7280;margin-top:4px}
.sample-card{background:#F8FAFC;border:1px solid #E5E7EB;border-radius:10px;padding:12px 14px;margin:8px 0}
</style>
""", unsafe_allow_html=True)

APP_VERSION = "v6.0-hybrid"
RF_OVERRIDE_DEFAULT = 85
MAX_UPLOAD_ROWS = 500000
MODEL_DIR = "saved_models"
REQUIRED_MODEL_FILES = ["isolation_forest.pkl", "scaler.pkl", "top_features.pkl", "label_encoder.pkl", "random_forest.pkl", "ae_threshold.pkl"]
OPTIONAL_MODEL_FILES = ["autoencoder.keras"]
REQUIREMENTS_TXT = """streamlit
pandas
numpy
joblib
plotly
scikit-learn
tensorflow
reportlab
"""

MODEL_PERFORMANCE_DATA = [
    {"Model":"Isolation Forest","Accuracy":0.8267,"Precision":0.4865,"Recall":0.4865,"F1":0.4865,"Not":"Unsupervised binary anomali değerlendirmesi"},
    {"Model":"Autoencoder","Accuracy":0.7523,"Precision":0.9173,"Recall":0.5546,"F1":0.6913,"Not":"Normal trafikle eğitilmiş reconstruction error yaklaşımı"},
    {"Model":"Random Forest","Accuracy":0.8599,"Precision":0.5449,"Recall":0.9973,"F1":0.7047,"Not":"Supervised sınıflandırma; RF tahmini destekleyici bilgi sağlar"},
]

LABEL_MAP = {
    "benign": ("Normal Trafik", "Normal"),
    "ftp-patator": ("FTP Brute Force", "Brute Force"),
    "ssh-patator": ("SSH Brute Force", "Brute Force"),
    "portscan": ("Port Tarama", "Reconnaissance"),
    "ddos": ("DDoS Trafiği", "DDoS"),
    "dos hulk": ("DoS Hulk", "DoS"),
    "dos goldeneye": ("DoS GoldenEye", "DoS"),
    "dos slowloris": ("DoS Slowloris", "DoS"),
    "dos slowhttptest": ("DoS SlowHTTPTest", "DoS"),
    "bot": ("Bot Trafiği", "Botnet"),
    "infiltration": ("Sızma / Infiltration", "Infiltration"),
    "heartbleed": ("Heartbleed Exploit", "Exploit"),
    "web attack brute force": ("Web Brute Force", "Web Attack"),
    "web attack xss": ("Web XSS", "Web Attack"),
    "web attack sql injection": ("Web SQL Injection", "Web Attack"),
}

risk_color_map = {"HIGH":"#E53935","MEDIUM":"#FFB300","LOW":"#43A047"}
category_color_map = {"Normal":"#43A047","DoS":"#E53935","DDoS":"#B71C1C","Brute Force":"#FB8C00","Reconnaissance":"#8E24AA","Web Attack":"#1E88E5","Botnet":"#6D4C41","Infiltration":"#546E7A","Exploit":"#D81B60","Diğer / Bilinmeyen":"#9E9E9E","N/A":"#BDBDBD"}

feature_defaults = {
    "Max Packet Length": (0.0, 65535.0, 512.0),
    "Packet Length Variance": (0.0, 100000.0, 100.0),
    "Average Packet Size": (0.0, 65535.0, 256.0),
    "Packet Length Mean": (0.0, 65535.0, 256.0),
    "Total Length of Fwd Packets": (0.0, 10000000.0, 1000.0),
    "Total Length of Bwd Packets": (0.0, 10000000.0, 500.0),
    "Flow Duration": (0.0, 120000000.0, 50000.0),
    "Total Fwd Packets": (0.0, 10000.0, 5.0),
    "Total Backward Packets": (0.0, 10000.0, 3.0),
}

# ──────────────────────────────────────────────────────────────
# Model loading and validation
# ──────────────────────────────────────────────────────────────
def validate_model_files():
    missing = [f for f in REQUIRED_MODEL_FILES if not os.path.exists(os.path.join(MODEL_DIR, f))]
    optional_missing = [f for f in OPTIONAL_MODEL_FILES if not os.path.exists(os.path.join(MODEL_DIR, f))]
    return missing, optional_missing

missing_model_files, optional_missing_model_files = validate_model_files()
if missing_model_files:
    st.error("Gerekli model dosyaları bulunamadı: " + ", ".join(missing_model_files))
    st.markdown("""
    <div class="warning-box"><b>Kurulum yardımı</b><br>
    Uygulamanın çalışması için <code>appv6.py</code> ile aynı klasörde <code>saved_models</code> dizini bulunmalıdır.
    Bu klasörde eğitim sürecinde oluşturulan model ve yardımcı dosyalar yer almalıdır.
    </div>
    """, unsafe_allow_html=True)
    with st.expander("Gerekli dosyalar ve çözüm adımları", expanded=True):
        st.markdown("""
        **Beklenen klasör yapısı**
        ```text
        BTRM/
        ├── appv6.py
        ├── requirements.txt
        └── saved_models/
            ├── isolation_forest.pkl
            ├── scaler.pkl
            ├── top_features.pkl
            ├── label_encoder.pkl
            ├── random_forest.pkl
            ├── ae_threshold.pkl
            └── autoencoder.keras
        ```

        **Çözüm adımları**
        1. `saved_models` klasörünün uygulama dosyasıyla aynı dizinde olduğundan emin olun.
        2. Eksik dosyalar varsa model eğitim notebook'unu tekrar çalıştırıp model dosyalarını yeniden üretin.
        3. Paketler eksikse terminalde `python -m pip install -r requirements.txt` komutunu çalıştırın.
        4. Uygulamayı `python -m streamlit run appv6.py` komutuyla başlatın.
        """)
    st.stop()

@st.cache_resource
def load_models():
    iso = joblib.load(os.path.join(MODEL_DIR, "isolation_forest.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    feats = list(joblib.load(os.path.join(MODEL_DIR, "top_features.pkl")))
    le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    rf = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
    ae_thr = joblib.load(os.path.join(MODEL_DIR, "ae_threshold.pkl"))
    ae = None
    ae_path = os.path.join(MODEL_DIR, "autoencoder.keras")
    if os.path.exists(ae_path):
        try:
            import tensorflow as tf
            ae = tf.keras.models.load_model(ae_path, compile=False)
        except Exception:
            ae = None
    return iso, scaler, feats, le, rf, ae_thr, ae

iso_model, scaler, top_features, le, rf_model, ae_threshold, ae_model = load_models()

# ──────────────────────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────────────────────
def normalize_label(label) -> str:
    """CSV içindeki label değerlerini güvenli ve standart hale getirir.
    Boş hücreler, NaN değerleri ve bozuk karakterler N/A kabul edilir.
    """
    if label is None:
        return "N/A"
    try:
        if pd.isna(label):
            return "N/A"
    except Exception:
        pass

    label = str(label).strip().replace("�", " ").replace("\ufffd", " ")
    label = " ".join(label.split())

    if label.lower() in ["", "nan", "none", "null", "n/a"]:
        return "N/A"

    return label

def clean_display_value(value, fallback="N/A") -> str:
    """Grafiklerde NaN/None gibi değerlerin görünmesini engeller."""
    if value is None:
        return fallback
    try:
        if pd.isna(value):
            return fallback
    except Exception:
        pass
    value = str(value).strip()
    if value.lower() in ["", "nan", "none", "null", "n/a"]:
        return fallback
    return value

def map_label(label):
    normalized = normalize_label(label)
    key = normalized.lower()
    if key in LABEL_MAP:
        return LABEL_MAP[key]
    if key == "n/a":
        return "N/A", "N/A"
    return normalized, "Diğer / Bilinmeyen"

def is_attack_label(label) -> bool:
    return normalize_label(label).lower() not in ["benign", "n/a", ""]

def has_valid_true_labels(results_df) -> bool:
    """Gerçek label bilgisi gerçekten dolu mu? Boş/NaN label kolonunu geçerli saymaz."""
    if "true_label" not in results_df.columns:
        return False
    labels = results_df["true_label"].apply(normalize_label)
    return labels.ne("N/A").any()

def get_chart_source_columns(results_df):
    """Grafiklerde gerçek label varsa onu, yoksa RF tahminlerini kaynak olarak kullanır."""
    if has_valid_true_labels(results_df):
        return "true_category", "true_display_label", "Gerçek Label"
    return "rf_category", "rf_display_label", "RF Tahmini"

def make_count_df(results_df, column, label_name, top_n=12):
    """Okunabilir grafikler için kategori sayım tablosu üretir.
    Çok fazla sınıf varsa düşük frekanslı değerleri 'Diğer' altında toplar.
    """
    if column not in results_df.columns or len(results_df) == 0:
        return pd.DataFrame({label_name: [], "Adet": []})

    series = results_df[column].apply(lambda x: clean_display_value(x, "N/A"))
    counts = series.value_counts(dropna=False)

    if len(counts) > top_n:
        top = counts.head(top_n - 1)
        other = counts.iloc[top_n - 1:].sum()
        counts = pd.concat([top, pd.Series({"Diğer": other})])

    out = counts.reset_index()
    out.columns = [label_name, "Adet"]
    return out.sort_values("Adet", ascending=True).reset_index(drop=True)

def apply_clean_count_axis(fig, max_value):
    """Adet eksenlerinin okunabilir kalması için dinamik eksen ayarı uygular."""
    fig.update_xaxes(rangemode="tozero", tickformat="d")
    if max_value <= 20:
        fig.update_xaxes(dtick=1)
    return fig

def safe_float(value, default=0.0):
    try:
        value = float(value)
        if np.isnan(value) or np.isinf(value):
            return default
        return value
    except Exception:
        return default

def color_risk(val):
    if val == "HIGH": return "background-color:#FFCDD2;color:#B71C1C;font-weight:bold"
    if val == "MEDIUM": return "background-color:#FFF3CD;color:#8A6D00;font-weight:bold"
    if val == "LOW": return "background-color:#C8E6C9;color:#1B5E20;font-weight:bold"
    return ""

def feature_comment(feature, value):
    value = safe_float(value)
    if feature == "Flow Duration":
        return "Çok kısa süreli akış" if value < 1000 else "Uzun süreli akış" if value > 1000000 else "Orta süreli akış"
    if feature in ["Total Fwd Packets", "Total Backward Packets"]:
        return "Paket yok" if value == 0 else "Düşük paket sayısı" if value <= 3 else "Yüksek paket sayısı" if value > 1000 else "Normal / orta paket sayısı"
    if feature in ["Max Packet Length", "Average Packet Size", "Packet Length Mean"]:
        return "Küçük paket boyutu" if value <= 64 else "Çok büyük paket boyutu" if value > 10000 else "Orta paket boyutu"
    if feature == "Packet Length Variance":
        return "Paket boyutu değişkenliği yok" if value == 0 else "Yüksek değişkenlik" if value > 50000 else "Orta değişkenlik"
    if feature in ["Total Length of Fwd Packets", "Total Length of Bwd Packets"]:
        return "Veri aktarımı yok" if value == 0 else "Yüksek veri hacmi" if value > 1000000 else "Orta veri hacmi"
    return "Feature değeri"

def build_scenarios():
    benign = {f:0.0 for f in top_features}
    benign.update({"Max Packet Length":64.0,"Average Packet Size":52.0,"Packet Length Mean":52.0,"Flow Duration":5000.0,"Total Fwd Packets":2.0,"Total Backward Packets":2.0})
    dos = {f:0.0 for f in top_features}
    dos.update({"Max Packet Length":65535.0,"Packet Length Variance":90000.0,"Average Packet Size":60000.0,"Packet Length Mean":60000.0,"Total Length of Fwd Packets":9000000.0,"Flow Duration":100.0,"Total Fwd Packets":5000.0})
    scan = {f:0.0 for f in top_features}
    scan.update({"Max Packet Length":40.0,"Average Packet Size":40.0,"Packet Length Mean":40.0,"Total Fwd Packets":1.0,"Total Backward Packets":0.0,"Flow Duration":200.0})
    high_volume = {f:0.0 for f in top_features}
    high_volume.update({"Max Packet Length":12000.0,"Packet Length Variance":60000.0,"Average Packet Size":8000.0,"Packet Length Mean":8000.0,"Total Length of Fwd Packets":3000000.0,"Total Length of Bwd Packets":2500000.0,"Flow Duration":2000000.0,"Total Fwd Packets":800.0,"Total Backward Packets":600.0})
    return {"Manuel Giriş":None,"🟢 Normal Trafik":benign,"🔴 DoS Benzeri Trafik":dos,"🟡 Port Tarama Benzeri Trafik":scan,"🟠 Yüksek Hacimli Şüpheli Trafik":high_volume}


def decide_hybrid_risk(if_anom: bool, ae_anom: bool, rf_label: str, rf_confidence: float, rf_override_enabled: bool=True, rf_override_threshold: float=RF_OVERRIDE_DEFAULT):
    """RF destekli hibrit karar motoru.

    IF ve AE anomali sinyali üretir. RF modeli sınıf tahmini ve yüksek güvenli
    saldırı durumlarında LOW kararını MEDIUM seviyesine yükselten destekleyici
    sinyal olarak kullanılır.
    """
    votes = int(bool(if_anom)) + int(bool(ae_anom))
    rf_attack = is_attack_label(rf_label)
    rf_override = bool(rf_override_enabled and rf_attack and float(rf_confidence) >= float(rf_override_threshold))

    if votes == 2:
        risk = "HIGH"
        reason = "IF ve AE birlikte anomali tespit etti"
    elif votes == 1:
        risk = "HIGH" if rf_override else "MEDIUM"
        reason = "Tek anomali sinyali + yüksek güvenli RF saldırı tahmini" if rf_override else "IF veya AE modellerinden biri anomali tespit etti"
    else:
        risk = "MEDIUM" if rf_override else "LOW"
        reason = "IF/AE normal; ancak RF yüksek güvenle saldırı tahmin ettiği için incelemeye yükseltildi" if rf_override else "IF/AE normal ve RF destekli yükseltme yok"

    if risk == "HIGH":
        emoji = "🔴"
        action = "Önerilen Aksiyon: Engelle ve olayı incelemeye al"
    elif risk == "MEDIUM":
        emoji = "🟡"
        action = "Önerilen Aksiyon: Analist incelemesine yönlendir"
    else:
        emoji = "🟢"
        action = "Önerilen Aksiyon: İzin ver"

    ifae_risk = "HIGH" if votes == 2 else "MEDIUM" if votes == 1 else "LOW"
    return risk, emoji, action, ifae_risk, rf_override, reason

def detect(flow: dict) -> dict:
    X = pd.DataFrame([flow])
    for col in top_features:
        if col not in X.columns:
            X[col] = 0.0
    X = X[top_features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_sc = scaler.transform(X)

    if_score = float(-iso_model.decision_function(X_sc)[0])
    if_anom = iso_model.predict(X_sc)[0] == -1

    rf_pred_enc = rf_model.predict(X_sc)[0]
    rf_label = le.inverse_transform([rf_pred_enc])[0]
    rf_proba = rf_model.predict_proba(X_sc)[0]
    rf_confidence = float(rf_proba.max()) * 100
    rf_display, rf_category = map_label(rf_label)

    ae_error, ae_anom = 0.0, False
    if ae_model is not None:
        recon = ae_model.predict(X_sc.astype(np.float32), verbose=0)
        ae_error = float(np.mean(np.square(X_sc - recon)))
        ae_anom = ae_error > ae_threshold

    risk, emoji, action, ifae_risk, rf_override, decision_reason = decide_hybrid_risk(
        if_anom, ae_anom, rf_label, rf_confidence, True, RF_OVERRIDE_DEFAULT
    )

    decision_detail = f"IF={'Anomali' if if_anom else 'Normal'} | AE={'Anomali' if ae_anom else 'Normal'} | RF={rf_display} (%{rf_confidence:.1f}) → {risk}"
    return {
        "timestamp": datetime.now().strftime("%H:%M:%S"), "risk": risk, "ifae_risk": ifae_risk, "rf_override": rf_override, "decision_reason": decision_reason, "emoji": emoji, "action": action,
        "if_score": round(if_score, 6), "if_anom": bool(if_anom),
        "rf_label": rf_label, "rf_display_label": rf_display, "rf_category": rf_category, "rf_confidence": round(rf_confidence, 1),
        "ae_error": round(ae_error, 6), "ae_anom": bool(ae_anom), "decision_detail": decision_detail,
        "model_mismatch": is_attack_label(rf_label) and ifae_risk == "LOW"
    }


def detect_batch(df_sample: pd.DataFrame, label_col=None, analysis_mode="N/A", batch_size=512, rf_override_enabled=True, rf_override_threshold=RF_OVERRIDE_DEFAULT) -> pd.DataFrame:
    """Toplu CSV analizi için batch inference yapar.

    Tek tek satır bazlı model çağırmak yerine scaler, Isolation Forest,
    Random Forest ve Autoencoder işlemlerini tek seferde uygular. Bu değişiklik
    model, threshold veya feature listesini değiştirmez; yalnızca hesaplama
    yöntemini hızlandırır.
    """
    if df_sample is None or df_sample.empty:
        return pd.DataFrame()

    X = df_sample.copy()
    for col in top_features:
        if col not in X.columns:
            X[col] = 0.0

    X_features = (
        X[top_features]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    X_sc = scaler.transform(X_features)

    if_scores = -iso_model.decision_function(X_sc)
    if_anoms = iso_model.predict(X_sc) == -1

    rf_pred_enc = rf_model.predict(X_sc)
    rf_labels = le.inverse_transform(rf_pred_enc)
    rf_probas = rf_model.predict_proba(X_sc)
    rf_confidences = rf_probas.max(axis=1) * 100

    if ae_model is not None:
        recon = ae_model.predict(X_sc.astype(np.float32), batch_size=batch_size, verbose=0)
        ae_errors = np.mean(np.square(X_sc - recon), axis=1)
        ae_anoms = ae_errors > ae_threshold
    else:
        ae_errors = np.zeros(len(df_sample), dtype=float)
        ae_anoms = np.zeros(len(df_sample), dtype=bool)

    votes = if_anoms.astype(int) + ae_anoms.astype(int)

    now = datetime.now().strftime("%H:%M:%S")
    records = []
    for i, (source_index, row) in enumerate(df_sample.iterrows()):
        rf_label = rf_labels[i]
        rf_display, rf_category = map_label(rf_label)
        risk, emoji, action, ifae_risk, rf_override, decision_reason = decide_hybrid_risk(
            bool(if_anoms[i]), bool(ae_anoms[i]), rf_label, float(rf_confidences[i]), rf_override_enabled, rf_override_threshold
        )

        if label_col:
            true_label = normalize_label(row.get(label_col, "N/A"))
            true_display, true_category = map_label(true_label)
        else:
            true_label, true_display, true_category = "N/A", "N/A", "N/A"

        r = {
            "timestamp": now,
            "risk": risk,
            "ifae_risk": ifae_risk,
            "rf_override": bool(rf_override),
            "decision_reason": decision_reason,
            "emoji": emoji,
            "action": action,
            "if_score": round(float(if_scores[i]), 6),
            "if_anom": bool(if_anoms[i]),
            "rf_label": rf_label,
            "rf_display_label": rf_display,
            "rf_category": rf_category,
            "rf_confidence": round(float(rf_confidences[i]), 1),
            "ae_error": round(float(ae_errors[i]), 6),
            "ae_anom": bool(ae_anoms[i]),
            "decision_detail": f"IF={'Anomali' if if_anoms[i] else 'Normal'} | AE={'Anomali' if ae_anoms[i] else 'Normal'} | RF={rf_display} (%{float(rf_confidences[i]):.1f}) → {risk}",
            "analysis_row": i + 1,
            "source_row": int(source_index) + 1 if str(source_index).isdigit() else str(source_index),
            "source_file": row.get("source_file", "N/A"),
            "true_label": true_label,
            "true_display_label": true_display,
            "true_category": true_category,
            "analysis_mode": analysis_mode,
        }
        r["model_mismatch"] = is_attack_label(r["rf_label"]) and r["ifae_risk"] == "LOW"
        r["feature_behavior"] = summarize_feature_behavior(row)
        rf_match, binary_status = compute_label_prediction_columns(r)
        r["rf_true_label_match"] = rf_match
        r["binary_status"] = binary_status
        r["label_comparison_note"] = compare_with_true_label(true_label, r["rf_label"], r["risk"])
        records.append(r)

    return pd.DataFrame(records)

def generate_decision_text(result):
    if result["risk"] == "HIGH":
        return "Bu akışta hem Isolation Forest hem Autoencoder anomali tespit etmiştir. Bu nedenle karar motoru HIGH risk üretmiştir."
    if result["risk"] == "MEDIUM":
        return "Bu akışta Isolation Forest veya Autoencoder modellerinden yalnızca biri anomali tespit etmiştir. Bu nedenle kayıt MEDIUM risk seviyesinde değerlendirilmiştir."
    if result.get("rf_override", False):
        return "IF ve AE anomali üretmemiş olsa da Random Forest yüksek güvenle saldırı sınıfı tahmin ettiği için kayıt MEDIUM seviyesinde incelemeye yükseltilmiştir."
    note = " Random Forest saldırı sınıfına yakın tahmin vermiştir; ancak güven eşiği aşılmadığı için nihai risk LOW kalmıştır." if is_attack_label(result.get("rf_label", "BENIGN")) else ""
    return "Bu akış için Isolation Forest ve Autoencoder anomali tespit etmemiştir. Bu nedenle karar motoru LOW risk üretmiştir." + note

def generate_analysis_report(result):
    return (f"Random Forest modeli akışı **{result['rf_display_label']}** sınıfına yakın tahmin etmiştir ve güven değeri **%{result['rf_confidence']}** olarak hesaplanmıştır. "
            f"Isolation Forest sonucu **{'anomali' if result['if_anom'] else 'normal'}**, Autoencoder sonucu **{'anomali' if result['ae_anom'] else 'normal'}** olarak değerlendirilmiştir. "
            f"Bu nedenle karar motoru akışı **{result['risk']}** risk seviyesinde sınıflandırmıştır. {result['action']}.")

def read_uploaded_csvs(uploaded_files, target_cols=None, max_rows=MAX_UPLOAD_ROWS, chunksize=50000):
    """CSV dosyalarını bellek dostu şekilde okur.

    Büyük CICIDS2017 dosyalarında tüm kolonları belleğe almak yerine yalnızca
    model için gereken feature kolonları, varsa Label kolonu ve source_file bilgisi tutulur.
    """
    frames = []
    loaded = 0
    target_cols = list(target_cols or top_features)
    keep_cols = set([c.strip() for c in target_cols] + ["Label"])
    for uploaded_file in uploaded_files:
        uploaded_file.seek(0)
        try:
            iterator = pd.read_csv(uploaded_file, chunksize=chunksize, usecols=lambda c: str(c).strip() in keep_cols)
            for chunk in iterator:
                chunk.columns = chunk.columns.str.strip()
                chunk["source_file"] = uploaded_file.name
                remaining = max_rows - loaded
                if remaining <= 0:
                    break
                if len(chunk) > remaining:
                    chunk = chunk.head(remaining)
                frames.append(chunk)
                loaded += len(chunk)
                if loaded >= max_rows:
                    break
        except ValueError:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)
            df.columns = df.columns.str.strip()
            cols = [c for c in df.columns if c in keep_cols]
            df = df[cols]
            df["source_file"] = uploaded_file.name
            remaining = max_rows - loaded
            if remaining <= 0:
                break
            if len(df) > remaining:
                df = df.head(remaining)
            frames.append(df)
            loaded += len(df)
        if loaded >= max_rows:
            break
    return pd.concat(frames, ignore_index=False) if frames else pd.DataFrame()

def prepare_sample(df, n, mode, label_col=None, selected_label=None, random_state=42):
    n = min(n, len(df))
    if n <= 0:
        return df.head(0).copy()
    if mode == "İlk N satır":
        return df.head(n).copy()
    if mode in ["Rastgele N satır", "Gerçek dağılıma yakın örneklem"]:
        return df.sample(n=n, random_state=random_state).copy()
    if label_col is None:
        return df.sample(n=n, random_state=random_state).copy() if len(df) >= n else df.copy()

    labels = df[label_col].astype(str).apply(normalize_label)

    if mode == "Sadece saldırı kayıtları":
        pool = df[labels.apply(is_attack_label)]
        return pool.sample(n=min(n, len(pool)), random_state=random_state).copy() if len(pool) else df.head(n).copy()

    if mode == "Seçili label" and selected_label is not None:
        pool = df[labels == selected_label]
        return pool.sample(n=min(n, len(pool)), random_state=random_state).copy() if len(pool) else df.head(n).copy()

    if mode == "Binary dengeli örneklem":
        benign_pool = df[~labels.apply(is_attack_label)]
        attack_pool = df[labels.apply(is_attack_label)]
        half = max(1, n // 2)
        parts = []
        if len(benign_pool):
            parts.append(benign_pool.sample(n=min(half, len(benign_pool)), random_state=random_state))
        if len(attack_pool):
            parts.append(attack_pool.sample(n=min(n - sum(len(p) for p in parts), len(attack_pool)), random_state=random_state))
        if parts:
            sample = pd.concat(parts)
            if len(sample) < n:
                remaining = df.drop(index=sample.index, errors="ignore")
                if len(remaining):
                    sample = pd.concat([sample, remaining.sample(n=min(n-len(sample), len(remaining)), random_state=random_state)])
            return sample.sample(n=min(n, len(sample)), random_state=random_state).copy()
        return df.sample(n=n, random_state=random_state).copy()

    if mode in ["Label dengeli örneklem", "Dengeli örneklem"]:
        temp = df.copy()
        temp["_lbl"] = labels
        groups = [g for _, g in temp.groupby("_lbl", dropna=False)]
        if not groups:
            return df.head(n).copy()
        per_group = max(1, n // len(groups))
        samples = [g.sample(n=min(len(g), per_group), random_state=random_state) for g in groups]
        sample = pd.concat(samples)
        if len(sample) < n:
            remaining = temp.drop(index=sample.index, errors="ignore")
            if len(remaining):
                sample = pd.concat([sample, remaining.sample(n=min(n-len(sample), len(remaining)), random_state=random_state)])
        if len(sample) > n:
            sample = sample.sample(n=n, random_state=random_state)
        return sample.drop(columns=["_lbl"], errors="ignore").copy()

    return df.head(n).copy()

def compute_binary_metrics(results_df, risk_col="risk"):
    """IF/AE risk motorunun gerçek BENIGN/ATTACK label bilgisiyle uyumunu hesaplar.
    Bu metrik Random Forest sınıf tahmin başarısı değildir.
    """
    if "true_label" not in results_df.columns:
        return None
    df = results_df[results_df["true_label"].apply(normalize_label) != "N/A"].copy()
    if len(df) == 0:
        return None
    df["actual_attack"] = df["true_label"].apply(is_attack_label)
    df["predicted_attack"] = df[risk_col].isin(["MEDIUM", "HIGH"])
    tp = int(((df.actual_attack == True) & (df.predicted_attack == True)).sum())
    tn = int(((df.actual_attack == False) & (df.predicted_attack == False)).sum())
    fp = int(((df.actual_attack == False) & (df.predicted_attack == True)).sum())
    fn = int(((df.actual_attack == True) & (df.predicted_attack == False)).sum())
    total = tp + tn + fp + fn
    return {
        "tp":tp, "tn":tn, "fp":fp, "fn":fn, "total":total,
        "accuracy": ((tp+tn)/total*100) if total else 0,
        "attack_detection": (tp/(tp+fn)*100) if (tp+fn) else 0,
        "false_positive_rate": (fp/(fp+tn)*100) if (fp+tn) else 0,
        "false_negative_rate": (fn/(fn+tp)*100) if (fn+tp) else 0,
        "confusion_df": pd.DataFrame({"Tahmin: Normal / LOW":[tn,fn],"Tahmin: Saldırı / MEDIUM-HIGH":[fp,tp]}, index=["Gerçek: Normal","Gerçek: Saldırı"]),
        "risk_col": risk_col
    }


def compute_rf_metrics(results_df):
    """Random Forest tahminlerinin gerçek label ile uyumunu ayrı hesaplar.
    RF tahmini, nihai IF/AE risk kararından bağımsız değerlendirilir.
    """
    if "true_label" not in results_df.columns or "rf_label" not in results_df.columns:
        return None
    df = results_df[results_df["true_label"].apply(normalize_label) != "N/A"].copy()
    if len(df) == 0:
        return None
    true_norm = df["true_label"].apply(lambda x: normalize_label(x).lower())
    rf_norm = df["rf_label"].apply(lambda x: normalize_label(x).lower())
    exact_match = (true_norm == rf_norm)
    actual_attack = df["true_label"].apply(is_attack_label)
    rf_attack = df["rf_label"].apply(is_attack_label)
    tp = int(((actual_attack == True) & (rf_attack == True)).sum())
    tn = int(((actual_attack == False) & (rf_attack == False)).sum())
    fp = int(((actual_attack == False) & (rf_attack == True)).sum())
    fn = int(((actual_attack == True) & (rf_attack == False)).sum())
    total = len(df)
    binary_accuracy = ((tp + tn) / total * 100) if total else 0
    exact_accuracy = (exact_match.sum() / total * 100) if total else 0
    attack_recall = (tp / (tp + fn) * 100) if (tp + fn) else 0
    return {
        "total": total,
        "exact_accuracy": exact_accuracy,
        "binary_accuracy": binary_accuracy,
        "attack_recall": attack_recall,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "confusion_df": pd.DataFrame({"RF Tahmin: Normal":[tn,fn],"RF Tahmin: Saldırı":[fp,tp]}, index=["Gerçek: Normal","Gerçek: Saldırı"])
    }

def generate_bulk_summary(results_df):
    """Dashboard üstündeki analiz özetini profesyonel ve yanıltıcı olmayan şekilde üretir."""
    total = len(results_df)
    high = int((results_df.risk == "HIGH").sum()) if total else 0
    med = int((results_df.risk == "MEDIUM").sum()) if total else 0
    low = int((results_df.risk == "LOW").sum()) if total else 0
    mismatch = int(results_df["model_mismatch"].sum()) if "model_mismatch" in results_df.columns else 0
    rf_overrides = int(results_df["rf_override"].sum()) if "rf_override" in results_df.columns else 0

    if has_valid_true_labels(results_df):
        valid = results_df[results_df["true_label"].apply(normalize_label) != "N/A"].copy()
        benign = int((~valid["true_label"].apply(is_attack_label)).sum())
        attack = int(valid["true_label"].apply(is_attack_label).sum())
        sample_mode = clean_display_value(valid["analysis_mode"].iloc[0] if "analysis_mode" in valid.columns and len(valid) else "N/A")
        data_text = (
            f"Bu örneklemde gerçek label bilgisine göre <b>{benign}</b> BENIGN ve <b>{attack}</b> ATTACK kayıt bulunmaktadır. "
            f"Kullanılan örnekleme modu <b>{sample_mode}</b> olarak görünmektedir. Örneklem dağılımı, seçilen moda bağlı olarak genel CICIDS2017 dağılımından farklı olabilir."
        )
    else:
        data_text = (
            "Bu analizde geçerli gerçek label bilgisi bulunmadığı için BENIGN/ATTACK dağılımı ve karar uyumu hesaplanmamıştır. "
            "Model yine tahmin ve risk seviyesi üretmiştir; ancak başarı oranı ölçülemez."
        )

    rf_top = results_df["rf_display_label"].apply(lambda x: clean_display_value(x, "N/A")).value_counts().idxmax() if total else "N/A"
    rf_text = f"Random Forest tahminlerinde en sık görülen sınıf <b>{rf_top}</b> olmuştur."

    if mismatch > 0:
        mismatch_text = (
            f"<b>{mismatch}</b> kayıtta Random Forest saldırı sınıfı tahmin ettiği halde IF/AE başlangıç kararı LOW seviyesinde kalmıştır. "
            f"RF destekli yumuşak karar motoru bu kayıtlardan <b>{rf_overrides}</b> tanesini MEDIUM/HIGH inceleme seviyesine yükseltmiştir."
        )
    else:
        mismatch_text = "Bu örneklemde Random Forest tahmini ile IF/AE başlangıç kararı arasında uyuşmazlık üreten kayıt bulunmamıştır."

    return f"""
    <div class="report-box">
      <b>Analiz Özeti</b><br><br>
      <b>Risk özeti:</b> Toplam <b>{total}</b> ağ akışı değerlendirilmiştir. RF destekli hibrit karar motoru <b>{high}</b> kaydı HIGH, <b>{med}</b> kaydı MEDIUM ve <b>{low}</b> kaydı LOW risk seviyesinde sınıflandırmıştır.<br><br>
      <b>Örneklem profili:</b> {data_text}<br><br>
      <b>RF tahmin özeti:</b> {rf_text}<br><br>
      <b>Model uyuşmazlığı:</b> {mismatch_text}
    </div>
    """

def render_stepper(current_step: int):
    steps = ["1. CSV yükle", "2. Ayarları seç", "3. Analizi başlat", "4. Sonuçları incele", "5. Raporu indir"]
    html = ['<div class="stepper">']
    for i, text in enumerate(steps, start=1):
        cls = "step done" if i < current_step else "step active" if i == current_step else "step"
        html.append(f'<div class="{cls}">{text}</div>')
    html.append('</div>')
    st.markdown("".join(html), unsafe_allow_html=True)


def render_hero_metrics(results_df):
    total = len(results_df)
    high = int((results_df.risk == "HIGH").sum())
    med = int((results_df.risk == "MEDIUM").sum())
    low = int((results_df.risk == "LOW").sum())
    mismatch = int(results_df.model_mismatch.sum()) if "model_mismatch" in results_df.columns else 0
    cards = [
        ("Toplam Akış", total, "Analiz edilen kayıt sayısı"),
        ("🔴 HIGH", high, "Güçlü risk / engelleme adayı"),
        ("🟡 MEDIUM", med, "Analist incelemesi adayı"),
        ("🟢 LOW", low, "Normal / düşük risk"),
        ("⚠️ Uyuşmazlık", mismatch, "RF ile IF/AE farklı yönde")
    ]
    cols = st.columns(5)
    for col, (label, value, hint) in zip(cols, cards):
        with col:
            st.markdown(f'<div class="hero-card"><div class="label">{label}</div><div class="value">{value}</div><div class="hint">{hint}</div></div>', unsafe_allow_html=True)


def render_sample_profile(results_df):
    st.markdown("#### Örneklem Profili")
    if not has_valid_true_labels(results_df):
        st.markdown('<div class="sample-card">Bu analiz etiketsiz veri üzerinde yapılmıştır. Model tahmin üretir; ancak gerçek başarı oranı ölçülemez.</div>', unsafe_allow_html=True)
        return
    valid = results_df[results_df["true_label"].apply(normalize_label) != "N/A"].copy()
    total = len(valid)
    attack = int(valid["true_label"].apply(is_attack_label).sum())
    benign = total - attack
    mode = clean_display_value(valid["analysis_mode"].iloc[0] if "analysis_mode" in valid.columns and total else "N/A")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Label'lı Kayıt", total)
    c2.metric("BENIGN", f"{benign} ({(benign/total*100 if total else 0):.1f}%)")
    c3.metric("ATTACK", f"{attack} ({(attack/total*100 if total else 0):.1f}%)")
    c4.metric("Örnekleme Modu", mode)
    if mode in ["Label dengeli örneklem", "Sadece saldırı kayıtları", "Seçili label"]:
        st.info("Bu örnekleme modu saldırı türlerini görünür kılmak için genel veri seti dağılımından farklı bir BENIGN/ATTACK oranı üretebilir. Accuracy yorumlanırken örneklem profili dikkate alınmalıdır.")

def render_terms_glossary(expanded=False):
    with st.expander("📘 Terimler Sözlüğü", expanded=expanded):
        st.markdown("""
        | Terim | Açıklama |
        |---|---|
        | **IF Skoru** | Isolation Forest modelinin anomalilik skorudur. Bu uygulamada skor yükseldikçe kayıt daha şüpheli kabul edilir. |
        | **AE Hatası** | Autoencoder modelinin giriş verisini yeniden oluştururken yaptığı hata değeridir. Eşik değeri aşarsa anomali kabul edilir. |
        | **RF Tahmini** | Random Forest modelinin kaydı hangi trafik/saldırı sınıfına benzettiğini gösterir. |
        | **Hibrit Karar Motoru Uyumu** | RF destekli nihai risk kararının gerçek BENIGN/ATTACK label bilgisiyle uyumunu gösterir. |
        | **RF Binary Başarısı** | Random Forest tahmininin normal/saldırı seviyesinde gerçek label ile uyumudur. |
        | **RF Sınıf Eşleşmesi** | Random Forest tahmininin gerçek label ile birebir aynı sınıfı tahmin etme oranıdır. |
        | **TP** | Gerçek saldırı olan bir kaydın MEDIUM/HIGH olarak yakalanmasıdır. |
        | **TN** | Gerçek normal olan bir kaydın LOW olarak değerlendirilmesidir. |
        | **FP** | Gerçek normal olan bir kaydın MEDIUM/HIGH olarak işaretlenmesidir. |
        | **FN** | Gerçek saldırı olan bir kaydın LOW olarak kalmasıdır. |
        | **Model Uyuşmazlığı** | RF saldırı sınıfı tahmin ettiği halde IF/AE başlangıç kararının LOW kalmasıdır. RF güveni eşik üzerindeyse kayıt MEDIUM seviyesine yükseltilebilir. |
        """)


def render_decision_flow_diagram():
    nodes = ["CSV / Manuel Girdi", "Feature Kontrolü", "Ölçekleme", "IF + AE + RF", "RF Destekli Hibrit Karar", "Dashboard / Rapor"]
    html = ['<div class="flow-diagram">']
    for i, node in enumerate(nodes):
        html.append(f'<div class="flow-node">{node}</div>')
        if i != len(nodes) - 1:
            html.append('<div class="flow-arrow">→</div>')
    html.append('</div>')
    st.markdown("".join(html), unsafe_allow_html=True)


def feature_unit(feature):
    if "Duration" in feature:
        return "µs"
    if "Length" in feature or "Size" in feature:
        return "byte"
    if "Packets" in feature:
        return "adet"
    if "Variance" in feature or "Mean" in feature:
        return "sayısal değer"
    return "değer"

def build_top_table(df, sort_col, columns, ascending=False, n=10):
    if sort_col not in df.columns: return pd.DataFrame()
    cols = [c for c in columns if c in df.columns]
    return df.sort_values(sort_col, ascending=ascending)[cols].head(n).copy() if cols else pd.DataFrame()

def render_top_table_or_info(df, sort_col, columns, message, ascending=False, n=10):
    """Top-N tabloyu güvenli gösterir; boş sonuçta teknik Streamlit nesnesi basılmaz."""
    table = build_top_table(df, sort_col, columns, ascending=ascending, n=n)
    if len(table) > 0:
        st.dataframe(table, width="stretch", hide_index=True)
    else:
        st.info(message)


def render_small_sample_dashboard(results_df, cat_col, type_col):
    """1-4 satırlık analizlerde grafiklere ek olarak okunabilir karar özeti gösterir."""
    st.markdown(
        '<div class="warning-box"><b>Az sayıda kayıt analizi:</b><br>'
        'Analiz edilen kayıt sayısı 5\'ten az olduğu için grafikler istatistiksel dağılım olarak değil, '
        'seçilen kayıtların görsel özeti olarak yorumlanmalıdır. Bu nedenle grafiklerin yanında karar özeti de gösterilmektedir.</div>',
        unsafe_allow_html=True
    )

    cols = [
        "source_file", "source_row", "true_label", "true_display_label", cat_col,
        "risk", "rf_display_label", "rf_confidence", "if_score", "if_anom",
        "ae_error", "ae_anom", "decision_detail", "binary_status", "label_comparison_note"
    ]
    cols = [c for c in cols if c in results_df.columns]
    summary = results_df[cols].copy()

    rename_map = {
        "source_file": "Dosya",
        "source_row": "Kaynak Satır",
        "true_label": "Gerçek Label",
        "true_display_label": "Okunabilir Tür",
        cat_col: "Kategori",
        "risk": "Nihai Risk",
        "rf_display_label": "RF Tahmini",
        "rf_confidence": "RF Güven (%)",
        "if_score": "IF Skoru",
        "if_anom": "IF Anomali",
        "ae_error": "AE Hatası",
        "ae_anom": "AE Anomali",
        "decision_detail": "Karar Detayı",
        "binary_status": "Binary Durum",
        "label_comparison_note": "Karşılaştırma Notu",
    }
    summary = summary.rename(columns=rename_map)
    if "Nihai Risk" in summary.columns:
        st.dataframe(summary.style.map(color_risk, subset=["Nihai Risk"]), width="stretch", hide_index=True)
    else:
        st.dataframe(summary, width="stretch", hide_index=True)

    if len(results_df) == 1:
        r = results_df.iloc[0]
        st.markdown("#### Tek Kayıt Karar Yorumu")
        if r.get("true_label", "N/A") != "N/A":
            label_text = f"Gerçek etiket <b>{r.get('true_display_label', r.get('true_label'))}</b> olarak verilmiştir."
        else:
            label_text = "Bu dosyada gerçek label bulunmadığı için yalnızca model tahmini yapılmıştır."
        st.markdown(
            f"""<div class=\"report-box\">
            {label_text}<br>
            Random Forest tahmini: <b>{r.get('rf_display_label','N/A')}</b> (%{r.get('rf_confidence','N/A')}).<br>
            Isolation Forest kararı: <b>{'Anomali' if r.get('if_anom', False) else 'Normal'}</b>.
            Autoencoder kararı: <b>{'Anomali' if r.get('ae_anom', False) else 'Normal'}</b>.<br>
            Nihai risk kararı <b>{r.get('risk','N/A')}</b> olarak üretilmiştir.
            </div>""",
            unsafe_allow_html=True
        )


def create_csv_template(include_label=True, include_example=False):
    """Uygulamanın beklediği kolonlarla CSV şablonu üretir."""
    cols = list(top_features)
    if include_label and "Label" not in cols:
        cols.append("Label")
    if include_example:
        row = {c: 0.0 for c in cols}
        for feat, (_, _, default) in feature_defaults.items():
            if feat in row:
                row[feat] = default
        if include_label:
            row["Label"] = "BENIGN"
        df = pd.DataFrame([row], columns=cols)
    else:
        df = pd.DataFrame(columns=cols)
    return df.to_csv(index=False).encode("utf-8-sig")

def summarize_feature_behavior(row, max_items=4):
    """Tek bir akış için seçilen top feature değerlerini kısa yorumlar halinde özetler."""
    notable = []
    priority_keywords = ["Çok", "Yüksek", "Paket yok", "Veri aktarımı yok", "Uzun", "kısa"]
    for feat in top_features:
        val = safe_float(row.get(feat, 0.0))
        comment = feature_comment(feat, val)
        if any(k.lower() in comment.lower() for k in priority_keywords):
            notable.append(f"{feat}: {comment} ({val:.3g})")
        if len(notable) >= max_items:
            break
    if not notable:
        return "Seçilen top feature değerlerinde belirgin uç davranış gözlenmedi."
    return "; ".join(notable)

def compare_with_true_label(true_label, rf_label, risk):
    """Gerçek label varsa RF sınıf tahmini ve binary risk kararını yorumlar."""
    if normalize_label(true_label) == "N/A":
        return "Label yok: yalnızca tahmin üretildi, performans karşılaştırması yapılmadı."
    true_norm = normalize_label(true_label).lower()
    rf_norm = normalize_label(rf_label).lower()
    actual_attack = is_attack_label(true_label)
    predicted_attack = risk in ["MEDIUM", "HIGH"]
    if not actual_attack and not predicted_attack:
        binary_note = "Binary karar: doğru normal (TN)."
    elif not actual_attack and predicted_attack:
        binary_note = "Binary karar: false positive; BENIGN kayıt şüpheli/riskli işaretlendi."
    elif actual_attack and predicted_attack:
        binary_note = "Binary karar: true positive; saldırı kaydı MEDIUM/HIGH yakalandı."
    else:
        binary_note = "Binary karar: false negative; saldırı kaydı LOW kaldı."
    rf_note = "RF sınıf tahmini gerçek label ile aynı." if true_norm == rf_norm else "RF sınıf tahmini gerçek label ile birebir aynı değil."
    return f"{binary_note} {rf_note}"

def compute_label_prediction_columns(row):
    true_label = row.get("true_label", "N/A")
    rf_label = row.get("rf_label", "BENIGN")
    risk = row.get("risk", "LOW")
    if normalize_label(true_label) == "N/A":
        return "N/A", "Label yok"
    rf_match = normalize_label(true_label).lower() == normalize_label(rf_label).lower()
    actual_attack = is_attack_label(true_label)
    predicted_attack = risk in ["MEDIUM", "HIGH"]
    if not actual_attack and not predicted_attack:
        binary_status = "TN - Doğru Normal"
    elif not actual_attack and predicted_attack:
        binary_status = "FP - Yanlış Alarm"
    elif actual_attack and predicted_attack:
        binary_status = "TP - Saldırı Yakalandı"
    else:
        binary_status = "FN - Saldırı LOW Kaldı"
    return ("Evet" if rf_match else "Hayır"), binary_status

def register_pdf_font():
    if not REPORTLAB_AVAILABLE: return None
    for path in ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/calibri.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"]:
        try:
            if os.path.exists(path):
                pdfmetrics.registerFont(TTFont("AppFont", path)); return "AppFont"
        except Exception:
            pass
    return "Helvetica"

def create_pdf_report(results_df, metrics=None):
    if not REPORTLAB_AVAILABLE: return None
    buffer = BytesIO(); font = register_pdf_font()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=1.3*cm, leftMargin=1.3*cm, topMargin=1.3*cm, bottomMargin=1.3*cm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="AppTitle", parent=styles["Title"], fontName=font, fontSize=18, leading=22))
    styles.add(ParagraphStyle(name="AppHeading", parent=styles["Heading2"], fontName=font, fontSize=13, leading=16, spaceAfter=8))
    styles.add(ParagraphStyle(name="AppBody", parent=styles["BodyText"], fontName=font, fontSize=9, leading=12))
    styles.add(ParagraphStyle(name="AppSmall", parent=styles["BodyText"], fontName=font, fontSize=7, leading=9))
    story = []
    total = len(results_df); high = int((results_df.risk == "HIGH").sum()); med = int((results_df.risk == "MEDIUM").sum()); low = int((results_df.risk == "LOW").sum())
    mismatch = int(results_df["model_mismatch"].sum()) if "model_mismatch" in results_df.columns else 0
    story += [Paragraph("Ağ Trafiği Anomali Tespit Raporu", styles["AppTitle"]), Spacer(1,10), Paragraph("Rapor zamanı: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"), styles["AppBody"]), Spacer(1,14)]
    story += [Paragraph("Yönetici Özeti", styles["AppHeading"]), Paragraph(f"Bu raporda toplam <b>{total}</b> ağ akışı analiz edilmiştir. RF destekli hibrit karar motoru sonucunda <b>{high}</b> HIGH, <b>{med}</b> MEDIUM ve <b>{low}</b> LOW riskli kayıt üretilmiştir. <b>{mismatch}</b> kayıt RF ile IF/AE başlangıç kararı arasındaki model uyuşmazlığı kapsamında değerlendirilmiştir.", styles["AppBody"]), Spacer(1,12)]
    rows = [["Risk Seviyesi","Kayıt Sayısı"],["HIGH", high],["MEDIUM", med],["LOW", low],["Model Uyuşmazlığı", mismatch]]
    if metrics is not None:
        rows += [["Hibrit Karar Motoru Uyumu", f"{metrics['accuracy']:.1f}%"],["Attack Detection Rate", f"{metrics['attack_detection']:.1f}%"],["False Positive", metrics["fp"]],["False Negative", metrics["fn"]]]
    table = Table(rows, colWidths=[8*cm, 5*cm])
    table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#E5E7EB")),("GRID",(0,0),(-1,-1),.5,colors.grey),("FONTNAME",(0,0),(-1,-1),font),("FONTSIZE",(0,0),(-1,-1),9),("ALIGN",(1,1),(-1,-1),"CENTER")]))
    story += [Paragraph("Risk ve Performans Özeti", styles["AppHeading"]), table, Spacer(1,14)]
    cat_col, _, _ = get_chart_source_columns(results_df)
    if cat_col in results_df.columns:
        cat_counts = make_count_df(results_df, cat_col, "Kategori", top_n=12).sort_values("Adet", ascending=False)
        cat_rows = [["Kategori","Adet"]] + cat_counts.values.tolist()
        cat_table = Table(cat_rows, colWidths=[8*cm, 5*cm])
        cat_table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#E5E7EB")),("GRID",(0,0),(-1,-1),.5,colors.grey),("FONTNAME",(0,0),(-1,-1),font),("FONTSIZE",(0,0),(-1,-1),9),("ALIGN",(1,1),(-1,-1),"CENTER")]))
        story += [Paragraph("Kategori Dağılımı", styles["AppHeading"]), cat_table, Spacer(1,14)]
    story += [Paragraph("Not", styles["AppHeading"]), Paragraph("Bu rapor akademik/prototip amaçlı IDS analiz çıktısıdır. Nihai risk kararı Isolation Forest ve Autoencoder karar motoruna göre üretilmiştir. Random Forest tahmini saldırı sınıfı hakkında destekleyici bilgi sağlar.", styles["AppBody"])]
    doc.build(story); buffer.seek(0); return buffer

# ──────────────────────────────────────────────────────────────
# Session and sidebar
# ──────────────────────────────────────────────────────────────
for key, val in {"log": [], "counters": {"HIGH":0,"MEDIUM":0,"LOW":0,"total":0}, "bulk_results": None, "bulk_file_key": None, "quick_scenario": "Manuel Giriş", "confirm_clear_log": False, "pdf_report_buffer": None}.items():
    if key not in st.session_state: st.session_state[key] = val

with st.sidebar:
    st.title("🛡️ IDS Dashboard")
    st.caption(f"Sürüm: {APP_VERSION}")
    st.divider()
    st.subheader("Model Durumu")
    st.write("Isolation Forest: ✅ Aktif")
    st.write("Random Forest: ✅ Aktif")
    st.write("Scaler: ✅ Aktif")
    st.write("Autoencoder: " + ("✅ Aktif" if ae_model is not None else "⚠️ Pasif"))
    st.metric("Top Feature Sayısı", len(top_features))
    st.divider()
    st.subheader("Çıktılar")
    st.write("CSV: ✅")
    st.write("PDF: " + ("✅" if REPORTLAB_AVAILABLE else "⚠️ reportlab yok"))
    st.divider()
    st.subheader("Oturum İstatistikleri")
    st.metric("Tekil analiz sayısı", st.session_state.counters.get("total", 0))
    st.caption(f"HIGH: {st.session_state.counters.get('HIGH',0)} · MEDIUM: {st.session_state.counters.get('MEDIUM',0)} · LOW: {st.session_state.counters.get('LOW',0)}")
    if st.session_state.get("bulk_results") is not None:
        st.metric("Son toplu analiz", len(st.session_state.bulk_results))
    else:
        st.caption("Toplu analiz henüz yapılmadı.")
    st.divider()
    st.subheader("Hızlı Senaryo")
    if st.button("🟢 Normal Trafik", width="stretch"):
        st.session_state.quick_scenario = "🟢 Normal Trafik"; st.rerun()
    if st.button("🔴 DoS Benzeri", width="stretch"):
        st.session_state.quick_scenario = "🔴 DoS Benzeri Trafik"; st.rerun()
    if st.button("🟡 Port Tarama", width="stretch"):
        st.session_state.quick_scenario = "🟡 Port Tarama Benzeri Trafik"; st.rerun()

st.title("🛡️ Ağ Trafiği Anomali Tespit Sistemi")
st.caption("CICIDS2017 veri setiyle eğitilmiş · RF destekli hibrit IDS karar motoru")
st.divider()

tab1, tab2, tab3 = st.tabs(["🔍 İnteraktif Akış Analizi", "📂 Toplu Trafik Analizi", "ℹ️ Sistem Mimarisi"])

# ──────────────────────────────────────────────────────────────
# Tab 1
# ──────────────────────────────────────────────────────────────
with tab1:
    st.subheader("İnteraktif Akış Analizi")
    st.markdown('<div class="info-box">Bu ekran, tek bir ağ akışı için model karar sürecini açıklanabilir şekilde gösterir.</div>', unsafe_allow_html=True)
    with st.expander("ℹ️ Bu ekran nasıl yorumlanır?", expanded=False):
        st.markdown("""
        - **Isolation Forest**, akışın normal trafikten ne kadar saptığını değerlendirir.
        - **Autoencoder**, akışı yeniden oluşturmaya çalışır; yüksek hata değeri anomali göstergesi olabilir.
        - **Random Forest**, akışın hangi trafik veya saldırı sınıfına benzediğini tahmin eder.
        - Nihai risk seviyesi **RF destekli hibrit karar motoru** ile belirlenir. IF ve AE anomali sinyali sağlar; RF yüksek güvenli saldırı tahmininde LOW kararını MEDIUM seviyesine yükseltebilir.
        """)
    col_form, col_result = st.columns([1,1], gap="large")
    with col_form:
        st.markdown("### Girdi Paneli")
        scenarios = build_scenarios()
        scenario_names = list(scenarios.keys())
        default_idx = scenario_names.index(st.session_state.quick_scenario) if st.session_state.quick_scenario in scenario_names else 0
        selected_scenario = st.selectbox("Analiz modu / hızlı senaryo", scenario_names, index=default_idx)
        st.session_state.quick_scenario = selected_scenario
        expectation_map = {
            "Manuel Giriş": "Manuel feature değerleriyle özel bir akış test edilir.",
            "🟢 Normal Trafik": "Genellikle LOW risk ve BENIGN/Normal Trafik tahmini beklenir.",
            "🔴 DoS Benzeri Trafik": "Yüksek hacim/uç değer davranışı nedeniyle MEDIUM veya HIGH risk beklenebilir.",
            "🟡 Port Tarama Benzeri Trafik": "Kısa süreli ve düşük paketli bağlantı davranışı test edilir.",
            "🟠 Yüksek Hacimli Şüpheli Trafik": "Veri hacmi ve paket istatistikleri yüksek bir akış senaryosu test edilir."
        }
        st.caption("Beklenen yorum: " + expectation_map.get(selected_scenario, "Seçilen senaryo analiz edilir."))
        if selected_scenario != "Manuel Giriş":
            flow_input = scenarios[selected_scenario]
            st.info("Seçilen senaryo değerleri otomatik olarak kullanılacak.")
            preview_df = pd.DataFrame([{"Feature":k,"Değer":v,"Yorum":feature_comment(k,v)} for k,v in flow_input.items() if v != 0.0])
            st.dataframe(preview_df, width="stretch", hide_index=True)
        else:
            flow_input = {f:0.0 for f in top_features}
            st.markdown('<div class="warning-box">Manuel giriş modunda yalnızca demo amaçlı temel feature değerleri değiştirilebilir. Diğer model feature değerleri 0 olarak gönderilir.</div>', unsafe_allow_html=True)
            with st.expander("⚙️ Gelişmiş Manuel Feature Girişi", expanded=True):
                for feat,(mn,mx,default) in feature_defaults.items():
                    if feat in top_features:
                        flow_input[feat] = st.number_input(f"{feat} ({feature_unit(feat)})", min_value=float(mn), max_value=float(mx), value=float(default), step=max(1.0, float(mx-mn)/1000), format="%.3f", help=f"Birim: {feature_unit(feat)}")
            manual_summary = pd.DataFrame([{"Feature":k,"Değer":v,"Yorum":feature_comment(k,v)} for k,v in flow_input.items() if k in feature_defaults])
            st.dataframe(manual_summary, width="stretch", hide_index=True)
        analyze = st.button("🔎 Akışı Analiz Et", type="primary", width="stretch")
        if analyze:
            st.success("Analiz tamamlandı. Karar özeti sağdaki panelde gösteriliyor.")
    with col_result:
        st.markdown("### Karar Paneli")
        if analyze:
            with st.spinner("Modeller çalışıyor..."):
                result = detect(flow_input)
            st.markdown(f'<div class="risk-{result["risk"].lower()}"><h3 style="margin:0">{result["emoji"]} {result["risk"]} RISK</h3><p style="margin:4px 0 0">{result["action"]}</p><small>🕐 {result["timestamp"]}</small><br><small>{result["decision_detail"]}</small></div>', unsafe_allow_html=True)
            st.markdown("#### Model Kararları")
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.markdown(f'<div class="model-card"><h4>Isolation Forest</h4><p><b>Sonuç:</b> {"⚠️ Anomali" if result["if_anom"] else "✅ Normal"}</p><p><b>IF Skoru:</b> {result["if_score"]}</p><p class="small-note">Skor yükseldikçe trafik daha şüpheli kabul edilir.</p></div>', unsafe_allow_html=True)
            with mc2:
                ae_status = "⚠️ Anomali" if result["ae_anom"] else "✅ Normal" if ae_model is not None else "Pasif"
                ae_thr = str(round(float(ae_threshold),6)) if ae_model is not None else "N/A"
                st.markdown(f'<div class="model-card"><h4>Autoencoder</h4><p><b>Sonuç:</b> {ae_status}</p><p><b>AE Hatası:</b> {result["ae_error"]}</p><p><b>Threshold:</b> {ae_thr}</p><p class="small-note">Hata threshold değerini aşarsa anomali kabul edilir.</p></div>', unsafe_allow_html=True)
            with mc3:
                st.markdown(f'<div class="model-card"><h4>Random Forest</h4><p><b>Tahmin:</b> {result["rf_display_label"]}</p><p><b>Kategori:</b> {result["rf_category"]}</p><p><b>Güven:</b> %{result["rf_confidence"]}</p><p class="small-note">Sınıf tahmini destekleyici bilgi sağlar.</p></div>', unsafe_allow_html=True)
            st.info(generate_decision_text(result))
            st.markdown(f'<div class="report-box">{generate_analysis_report(result)}</div>', unsafe_allow_html=True)
            if result.get("model_mismatch", False):
                if result.get("rf_override", False):
                    st.warning("Model Uyuşmazlığı: Random Forest yüksek güvenle saldırı tahmin ettiği için IF/AE LOW başlangıç kararı MEDIUM seviyesine yükseltildi.")
                else:
                    st.warning("Model Uyuşmazlığı: Random Forest saldırı sınıfı tahmin etti; ancak güven eşiği aşılmadığı için kayıt LOW seviyesinde kaldı.")
            st.session_state.log.append(result); st.session_state.counters[result["risk"]]+=1; st.session_state.counters["total"]+=1
        else:
            st.markdown('<div class="info-box">Analiz sonucu burada gösterilecektir. Sol taraftan bir senaryo seçin veya manuel feature değerleri girerek analizi başlatın.</div>', unsafe_allow_html=True)
        st.divider(); st.markdown("### Oturum İstatistikleri")
        m1,m2,m3,m4 = st.columns(4); m1.metric("Toplam", st.session_state.counters["total"]); m2.metric("🔴 Yüksek", st.session_state.counters["HIGH"]); m3.metric("🟡 Orta", st.session_state.counters["MEDIUM"]); m4.metric("🟢 Düşük", st.session_state.counters["LOW"])
        if st.session_state.log:
            log_df = pd.DataFrame(st.session_state.log)[["timestamp","risk","rf_display_label","rf_category","rf_confidence","if_score","ae_error","decision_detail","action"]]
            log_df.columns = ["Zaman","Risk","RF Tahmini","RF Kategorisi","RF Güven (%)","IF Skoru","AE Hatası","Karar Detayı","Önerilen Aksiyon"]
            st.dataframe(log_df.style.map(color_risk, subset=["Risk"]), width="stretch", hide_index=True)
            if not st.session_state.confirm_clear_log:
                if st.button("🗑️ Logu Temizle", width="stretch"):
                    st.session_state.confirm_clear_log = True
                    st.rerun()
            else:
                st.warning("Oturum logu temizlenecek. Bu işlem geri alınamaz.")
                c_yes, c_no = st.columns(2)
                if c_yes.button("Evet, temizle", width="stretch"):
                    st.session_state.log=[]; st.session_state.counters={"HIGH":0,"MEDIUM":0,"LOW":0,"total":0}; st.session_state.confirm_clear_log=False; st.rerun()
                if c_no.button("Vazgeç", width="stretch"):
                    st.session_state.confirm_clear_log=False; st.rerun()

# ──────────────────────────────────────────────────────────────
# Tab 2
# ──────────────────────────────────────────────────────────────
with tab2:
    st.subheader("CSV Dosyası Yükle & Toplu Trafik Analizi")
    st.markdown('<div class="info-box">CICIDS2017 formatında bir veya birden fazla CSV yükleyin. Modelin çalışması için Label kolonu zorunlu değildir; Label varsa yalnızca gerçek label bazlı performans karşılaştırması için kullanılır. Label kolonu model girişine verilmez.</div>', unsafe_allow_html=True)
    st.markdown("#### CSV Şablonu")
    st.caption("Veri yüklerken aşağıdaki şablondaki feature kolonlarını kullanın. Label opsiyoneldir; etiketsiz dosyada tahmin yapılır fakat performans ölçümü yapılamaz.")
    tmpl1, tmpl2 = st.columns(2)
    with tmpl1:
        st.download_button("📄 Boş CSV Şablonu İndir", create_csv_template(include_label=True, include_example=False), "ids_bos_csv_sablonu.csv", "text/csv", width="stretch")
    with tmpl2:
        st.download_button("🧪 Örnek Tek Satır CSV İndir", create_csv_template(include_label=True, include_example=True), "ids_ornek_tek_satir.csv", "text/csv", width="stretch")
    render_terms_glossary(expanded=False)
    uploaded_files = st.file_uploader("CSV Dosyası Seç", type=["csv"], accept_multiple_files=True)
    if uploaded_files:
        st.markdown('<div class="cta-box"><b>Dosya başarıyla seçildi.</b><br>Şimdi analiz ayarlarını kontrol edin ve aşağıdaki <b>Toplu Analiz Başlat</b> butonuyla işlemi başlatın.</div>', unsafe_allow_html=True)
        file_key = "|".join([f"{f.name}:{getattr(f,'size',0)}" for f in uploaded_files])
        if st.session_state.bulk_file_key != file_key:
            st.session_state.bulk_results = None
            st.session_state.pdf_report_buffer = None
            st.session_state.bulk_file_key = file_key
        try:
            df_up = read_uploaded_csvs(uploaded_files, target_cols=top_features)
        except Exception as e:
            st.error("CSV dosyası okunurken hata oluştu: " + str(e)); st.stop()
        if df_up.empty: st.error("Yüklenen CSV dosyası boş görünüyor."); st.stop()
        df_up.columns = df_up.columns.str.strip()
        st.write(f"**{len(df_up)} satır** yüklendi. İlk 5 satır:"); st.dataframe(df_up.head(), width="stretch")
        if len(df_up) >= MAX_UPLOAD_ROWS:
            st.warning(f"Performans ve bellek güvenliği için en fazla {MAX_UPLOAD_ROWS:,} satır belleğe alınmıştır. Daha büyük analizler için örnekleme modlarını kullanın.")
        with st.expander("📁 Yüklenen Dosya Özeti", expanded=False):
            src = df_up["source_file"].value_counts().reset_index(); src.columns=["Dosya","Satır Sayısı"]; st.dataframe(src, width="stretch", hide_index=True)
        label_col = next((c for c in df_up.columns if c.strip().lower()=="label"), None)
        missing = [f for f in top_features if f not in df_up.columns]
        if missing: st.error("Eksik featurelar: " + str(missing)); st.stop()
        if label_col:
            lp = df_up[label_col].astype(str).apply(normalize_label)
            ldf = lp.value_counts().reset_index(); ldf.columns=["Orijinal Label","Satır Sayısı"]
            ldf["Okunabilir Tür"] = ldf["Orijinal Label"].apply(lambda x: map_label(x)[0]); ldf["Kategori"] = ldf["Orijinal Label"].apply(lambda x: map_label(x)[1])
            with st.expander("🏷️ Yüklenen CSV Label Özeti", expanded=False): st.dataframe(ldf, width="stretch", hide_index=True)
            st.info("Label kolonu yalnızca değerlendirme ve karşılaştırma için kullanılır. Model tahmininde kullanılan feature listesine Label dahil edilmez; bu nedenle etiket sızıntısı/label leakage engellenmiştir.")
        else:
            st.warning("CSV içinde Label kolonu bulunamadı. Model yine tahmin üretecektir; ancak gerçek label bazlı performans özeti, confusion matrix ve label karşılaştırması hesaplanamayacaktır.")
        st.divider()
        st.markdown("### 2. Analiz Ayarları")
        st.caption("Varsayılan ayarlar çoğu test için yeterlidir. Gelişmiş seçenekleri yalnızca belirli senaryoları test etmek istediğinizde değiştirin.")
        modes = ["İlk N satır", "Rastgele N satır"] + (["Gerçek dağılıma yakın örneklem", "Binary dengeli örneklem", "Label dengeli örneklem", "Sadece saldırı kayıtları", "Seçili label"] if label_col else [])
        with st.expander("⚙️ Analiz Ayarları", expanded=True):
            c1,c2,c3 = st.columns([2,1,1])
            analysis_mode = c1.selectbox("Analiz modu", modes, help="Verinin hangi örnekleme yöntemiyle analiz edileceğini belirler.")
            max_rows = min(10000, len(df_up))
            if max_rows == 1:
                with c2:
                    st.metric("Satır sayısı", 1)
                n_analyze = 1
            else:
                n_analyze = c2.slider("Satır sayısı", min_value=1, max_value=max_rows, value=min(500, len(df_up)), step=1, help="Analiz edilecek maksimum kayıt sayısı.")
            random_state = c3.number_input("Random seed", min_value=1, max_value=9999, value=42, step=1, help="Rastgele örneklem seçiminin tekrar üretilebilir olmasını sağlar.")
            selected_label = None
            if analysis_mode == "Seçili label" and label_col:
                labels = sorted(df_up[label_col].astype(str).apply(normalize_label).unique().tolist())
                selected_label = st.selectbox("Analiz edilecek label", labels, format_func=lambda x: f"{map_label(x)[0]} | Orijinal: {x}")
            st.markdown("**RF destekli yumuşak karar motoru**")
            k1, k2 = st.columns([1, 1])
            rf_override_enabled = k1.checkbox("RF yüksek güvenli saldırı tahmininde LOW kararı MEDIUM'a yükselt", value=True, help="False negative riskini azaltmak için RF saldırı tahmini yüksek güvenliyse kayıt analist incelemesine alınır.")
            rf_override_threshold = k2.slider("RF güven eşiği (%)", min_value=50, max_value=99, value=RF_OVERRIDE_DEFAULT, step=1, help="RF güveni bu eşik veya üzerinde ise destekli yükseltme uygulanır.")
        st.markdown("### 3. Analizi Başlat")
        st.caption("Performans için model tahminleri satır satır değil, batch modda toplu şekilde çalıştırılır. Bu işlem model doğruluğunu değiştirmez.")
        if st.button("🚀 Toplu Analiz Başlat", type="primary", width="stretch"):
            df_sample = prepare_sample(df_up, n_analyze, analysis_mode, label_col, selected_label, int(random_state))
            if df_sample.empty: st.error("Seçilen analiz moduna göre analiz edilecek kayıt bulunamadı."); st.stop()
            progress = st.progress(0)
            status_box = st.empty()

            status_box.info("1/4 Örneklem hazırlanıyor...")
            progress.progress(20)

            status_box.info("2/4 Modeller batch modda çalıştırılıyor...")
            progress.progress(45)
            results_df_fast = detect_batch(df_sample, label_col=label_col, analysis_mode=analysis_mode, batch_size=512, rf_override_enabled=rf_override_enabled, rf_override_threshold=rf_override_threshold)

            status_box.info("3/4 Sonuçlar ve karşılaştırma alanları hazırlanıyor...")
            progress.progress(85)
            st.session_state.bulk_results = results_df_fast
            st.session_state.pdf_report_buffer = None

            progress.progress(100)
            status_box.success("4/4 Analiz tamamlandı. Sonuçlar aşağıda gösteriliyor.")
        if st.session_state.bulk_results is not None:
            results_df = st.session_state.bulk_results.copy()
            st.success(f"✅ {len(results_df)} akış analiz edildi")
            render_hero_metrics(results_df)
            render_sample_profile(results_df)
            st.markdown(generate_bulk_summary(results_df), unsafe_allow_html=True)
            high = int((results_df.risk=="HIGH").sum()); med=int((results_df.risk=="MEDIUM").sum()); low=int((results_df.risk=="LOW").sum()); mismatch=int(results_df.model_mismatch.sum())
            binary_metrics = compute_binary_metrics(results_df, risk_col="risk")
            ifae_binary_metrics = compute_binary_metrics(results_df, risk_col="ifae_risk") if "ifae_risk" in results_df.columns else None
            rf_metrics = compute_rf_metrics(results_df)
            if binary_metrics:
                st.divider(); st.subheader("🎯 Etiketli Veri Üzerinde Değerlendirme")
                st.markdown(
                    '<div class="info-box"><b>Önemli ayrım:</b><br>'
                    'Buradaki <b>Hibrit Karar Motoru Uyumu</b>, yüklenen veri üzerinde RF destekli nihai risk kararının gerçek BENIGN/ATTACK label bilgisiyle uyumunu gösterir. '
                    'Random Forest sınıflandırma başarısı ve IF/AE başlangıç kararı ayrı değerlendirilir.</div>',
                    unsafe_allow_html=True
                )
                p1,p2,p3,p4 = st.columns(4)
                p1.metric("Hibrit Karar Motoru Uyumu", f"{binary_metrics['accuracy']:.1f}%", help="LOW=normal, MEDIUM/HIGH=saldırı/şüpheli kabul edilerek RF destekli nihai karar için hesaplanır.")
                if rf_metrics:
                    p2.metric("RF Binary Başarısı", f"{rf_metrics['binary_accuracy']:.1f}%", help="Random Forest tahmininin BENIGN/ATTACK seviyesinde gerçek label ile uyumudur.")
                    p3.metric("RF Sınıf Eşleşmesi", f"{rf_metrics['exact_accuracy']:.1f}%", help="Random Forest tahmininin gerçek label ile birebir aynı sınıfı tahmin etme oranıdır.")
                else:
                    p2.metric("RF Binary Başarısı", "Ölçülemedi")
                    p3.metric("RF Sınıf Eşleşmesi", "Ölçülemedi")
                p4.metric("False Negative (Hibrit)", binary_metrics["fn"], f"{binary_metrics['false_negative_rate']:.1f}%", help="Gerçek saldırı kayıtlarının nihai hibrit karar motorunda LOW kalmasıdır.")

                m1,m2,m3,m4 = st.columns(4)
                m1.metric("Attack Detection Rate (Hibrit)", f"{binary_metrics['attack_detection']:.1f}%", f"TP: {binary_metrics['tp']}", help="Gerçek saldırı kayıtlarının nihai hibrit karar motorunda MEDIUM/HIGH olarak yakalanma oranıdır.")
                m2.metric("False Positive (Hibrit)", binary_metrics["fp"], f"{binary_metrics['false_positive_rate']:.1f}%", help="Gerçek BENIGN kayıtların nihai hibrit karar motorunda riskli işaretlenmesidir.")
                if rf_metrics:
                    m3.metric("RF Attack Recall", f"{rf_metrics['attack_recall']:.1f}%", help="Random Forest'ın gerçek saldırı kayıtlarını BENIGN dışı sınıflara atama oranıdır.")
                else:
                    m3.metric("RF Attack Recall", "Ölçülemedi")
                m4.metric("Değerlendirilen Label'lı Kayıt", binary_metrics["total"])

                with st.expander("Confusion Matrix Detayları", expanded=False):
                    c_if, c_rf = st.columns(2)
                    with c_if:
                        st.markdown("**Hibrit Karar Motoru**")
                        st.dataframe(binary_metrics["confusion_df"], width="stretch")
                    with c_rf:
                        st.markdown("**Random Forest Binary Değerlendirme**")
                        if rf_metrics:
                            st.dataframe(rf_metrics["confusion_df"], width="stretch")
                        else:
                            st.info("RF değerlendirmesi için yeterli label bilgisi bulunamadı.")
                render_terms_glossary(expanded=False)
            else:
                st.divider(); st.subheader("🎯 Etiketsiz Tahmin Modu")
                st.markdown(
                    '<div class="info-box"><b>Bu veri etiketsiz analiz edilmiştir.</b><br>'
                    'Model tahmin ve risk seviyesi üretmiştir; ancak dosyada geçerli <b>Label</b> bilgisi olmadığı için accuracy, false positive, false negative ve attack detection rate gibi gerçek başarı metrikleri hesaplanamaz. '
                    'Etiketsiz veri modelin başarısız olduğu anlamına gelmez; yalnızca tahminlerin doğruluğu ölçülemez.</div>',
                    unsafe_allow_html=True
                )
            st.divider(); st.subheader("📊 Analiz Dashboard")
            cat_col, type_col, chart_source_label = get_chart_source_columns(results_df)

            st.markdown(
                '<div class="info-box"><b>Dashboard yorumlama notu:</b><br>'
                'Grafikler her analizde gösterilir. Kayıt sayısı az olduğunda grafikler genel dağılımı değil, seçilen kayıtların özetini gösterir. '
                'Bu nedenle tek satır veya az satırlı analizlerde grafiklerle birlikte karar özeti de sunulur.</div>',
                unsafe_allow_html=True
            )

            if len(results_df) < 5:
                render_small_sample_dashboard(results_df, cat_col, type_col)

            ch1,ch2 = st.columns(2)
            with ch1:
                rc = results_df.risk.value_counts().reindex(["HIGH","MEDIUM","LOW"]).fillna(0).reset_index()
                rc.columns=["Risk","Adet"]
                fig = px.pie(
                    rc,
                    names="Risk",
                    values="Adet",
                    title="Risk Dağılımı",
                    hole=.45,
                    color="Risk",
                    color_discrete_map=risk_color_map
                )
                fig.update_traces(textposition="inside", textinfo="percent+label")
                fig.update_layout(height=380, margin=dict(l=20,r=20,t=55,b=20), font=dict(size=12))
                st.plotly_chart(fig, width="stretch")

            with ch2:
                cc = make_count_df(results_df, cat_col, "Kategori", top_n=12)
                fig = px.bar(
                    cc,
                    x="Adet",
                    y="Kategori",
                    orientation="h",
                    title=f"Saldırı / Trafik Kategorisi Dağılımı ({chart_source_label})",
                    text="Adet",
                    color="Kategori",
                    color_discrete_map=category_color_map
                )
                fig.update_traces(textposition="outside", cliponaxis=False)
                fig.update_layout(
                    height=max(340, 34 * max(len(cc), 1) + 160),
                    showlegend=False,
                    xaxis_title="Akış Sayısı",
                    yaxis_title="",
                    margin=dict(l=20,r=60,t=55,b=35),
                    font=dict(size=12)
                )
                if len(cc) > 0:
                    apply_clean_count_axis(fig, int(cc["Adet"].max()))
                st.plotly_chart(fig, width="stretch")

            ch3,ch4 = st.columns(2)
            with ch3:
                tc = make_count_df(results_df, type_col, "Trafik / Saldırı Türü", top_n=12)
                fig = px.bar(
                    tc,
                    x="Adet",
                    y="Trafik / Saldırı Türü",
                    orientation="h",
                    title=f"Detaylı Trafik / Saldırı Türü Dağılımı ({chart_source_label})",
                    text="Adet"
                )
                fig.update_traces(textposition="outside", cliponaxis=False)
                fig.update_layout(
                    height=max(360, 36 * max(len(tc), 1) + 170),
                    xaxis_title="Akış Sayısı",
                    yaxis_title="",
                    margin=dict(l=20,r=70,t=55,b=35),
                    font=dict(size=12)
                )
                if len(tc) > 0:
                    apply_clean_count_axis(fig, int(tc["Adet"].max()))
                st.plotly_chart(fig, width="stretch")

            with ch4:
                rf = make_count_df(results_df, "rf_display_label", "RF Tahmini", top_n=12)
                fig = px.bar(
                    rf,
                    x="Adet",
                    y="RF Tahmini",
                    orientation="h",
                    title="Random Forest Tahmin Dağılımı",
                    text="Adet"
                )
                fig.update_traces(textposition="outside", cliponaxis=False)
                fig.update_layout(
                    height=max(360, 36 * max(len(rf), 1) + 170),
                    xaxis_title="Akış Sayısı",
                    yaxis_title="",
                    margin=dict(l=20,r=70,t=55,b=35),
                    font=dict(size=12)
                )
                if len(rf) > 0:
                    apply_clean_count_axis(fig, int(rf["Adet"].max()))
                st.plotly_chart(fig, width="stretch")

            ch5,ch6 = st.columns(2)
            with ch5:
                n_bins = min(20, max(4, int(np.sqrt(max(len(results_df), 1)))))
                fig = px.histogram(
                    results_df,
                    x="if_score",
                    color="risk",
                    nbins=n_bins,
                    title="Isolation Forest Skor Dağılımı",
                    color_discrete_map=risk_color_map
                )
                fig.update_layout(
                    height=380,
                    xaxis_title="IF Skoru",
                    yaxis_title="Akış Sayısı",
                    bargap=0.05,
                    margin=dict(l=20,r=20,t=55,b=40),
                    font=dict(size=12)
                )
                fig.update_yaxes(rangemode="tozero", tickformat="d")
                st.plotly_chart(fig, width="stretch")

            with ch6:
                dc = make_count_df(results_df, "decision_detail", "Karar Detayı", top_n=8)
                fig = px.bar(
                    dc,
                    x="Adet",
                    y="Karar Detayı",
                    orientation="h",
                    title="IF / AE Karar Motoru Dağılımı",
                    text="Adet"
                )
                fig.update_traces(textposition="outside", cliponaxis=False)
                fig.update_layout(
                    height=max(340, 38 * max(len(dc), 1) + 160),
                    xaxis_title="Akış Sayısı",
                    yaxis_title="",
                    margin=dict(l=20,r=70,t=55,b=35),
                    font=dict(size=12)
                )
                if len(dc) > 0:
                    apply_clean_count_axis(fig, int(dc["Adet"].max()))
                st.plotly_chart(fig, width="stretch")

            if len(results_df) < 5:
                st.caption("Not: Az sayıda kayıt analizinde grafiklerde görülen adetler yalnızca seçilen kayıtları temsil eder; genel model performansı veya veri seti dağılımı olarak yorumlanmamalıdır.")

            st.divider(); st.subheader("🚨 En Şüpheli Akışlar")
            st.caption("Bu bölüm; davranışsal anomali skoru, Autoencoder hatası ve RF güveni açısından dikkat çeken kayıtları ayrı ayrı listeler.")
            top_cols = ["source_file","source_row","true_display_label","true_category","risk","rf_display_label","rf_confidence","if_score","ae_error","decision_detail"]
            t1,t2,t3 = st.tabs(["En Yüksek IF Skoru", "En Yüksek AE Hatası", "En Yüksek RF Güveni"])
            with t1:
                render_top_table_or_info(results_df, "if_score", top_cols, "IF skoru için gösterilecek kayıt bulunamadı.", ascending=False, n=10)
            with t2:
                render_top_table_or_info(results_df, "ae_error", top_cols, "AE hatası için gösterilecek kayıt bulunamadı.", ascending=False, n=10)
            with t3:
                rf_attack = results_df[results_df.rf_label.apply(is_attack_label)].copy()
                render_top_table_or_info(rf_attack, "rf_confidence", top_cols, "RF tarafından saldırı sınıfı tahmin edilen kayıt bulunamadı.", ascending=False, n=10)
            st.divider(); st.subheader("🧬 Feature ve Label Karşılaştırması")
            feature_cols = ["source_file","source_row","true_label","true_display_label","risk","rf_display_label","rf_true_label_match","binary_status","feature_behavior","label_comparison_note"]
            feature_cols = [c for c in feature_cols if c in results_df.columns]
            feature_show = results_df[feature_cols].head(100).copy()
            feature_show.columns = ["Dosya","Kaynak Satır","Gerçek Label","Okunabilir Tür","Risk","RF Tahmini","RF-Label Eşleşti mi?","Binary Durum","Feature Davranış Özeti","Karşılaştırma Notu"][:len(feature_show.columns)]
            st.dataframe(feature_show, width="stretch", hide_index=True)
            st.caption("İlk 100 kayıt gösterilir. Bu bölüm, modelin kullandığı feature davranışı ile gerçek label ve RF tahmini arasındaki ilişkiyi yorumlamak için eklenmiştir.")

            if mismatch > 0:
                st.divider(); st.subheader("⚠️ Model Uyuşmazlığı Kayıtları")
                st.markdown('<div class="warning-box"><b>Model uyuşmazlığı ne demek?</b><br>Random Forest bir saldırı sınıfı tahmin ettiği halde IF/AE başlangıç kararı LOW kalmışsa bu kayıt model uyuşmazlığıdır. RF güveni eşik üzerindeyse hibrit motor bu kaydı MEDIUM seviyesine yükseltir; değilse analist incelemesine aday olarak listelenir.</div>', unsafe_allow_html=True)
                mm = results_df[results_df.model_mismatch == True][["source_file","source_row","true_label","true_display_label","true_category","risk","rf_display_label","rf_category","rf_confidence","decision_detail","label_comparison_note"]].head(100).copy()
                mm.columns = ["Dosya","Kaynak Satır","Gerçek Label","Okunabilir Tür","Kategori","Risk","RF Tahmini","RF Kategorisi","RF Güven (%)","Karar Detayı","Açıklama"]
                st.dataframe(mm, width="stretch", hide_index=True)
            st.divider(); st.subheader("🔎 Detaylı Sonuçlar")
            filtered = results_df.copy()
            with st.expander("🔧 Gelişmiş Filtreler", expanded=False):
                st.caption("Varsayılan olarak tüm kayıtlar gösterilir. Belirli bir risk, kategori, RF tahmini, dosya veya model uyuşmazlığı incelemek için filtreleri kullanın.")
                f1,f2,f3,f4 = st.columns(4)
                sr = f1.selectbox("Risk filtresi", ["Tümü"]+sorted(results_df.risk.apply(lambda x: clean_display_value(x, "N/A")).unique().tolist()))
                category_filter_series = results_df[cat_col].apply(lambda x: clean_display_value(x, "N/A"))
                rf_filter_series = results_df.rf_display_label.apply(lambda x: clean_display_value(x, "N/A"))
                sc = f2.selectbox("Kategori filtresi", ["Tümü"]+sorted(category_filter_series.unique().tolist()))
                sf = f3.selectbox("RF tahmini filtresi", ["Tümü"]+sorted(rf_filter_series.unique().tolist()))
                only_mismatch = f4.checkbox("Sadece model uyuşmazlığı")
                ss = st.selectbox("Dosya filtresi", ["Tümü"]+sorted(results_df.source_file.astype(str).unique().tolist()))
            if sr != "Tümü": filtered = filtered[filtered.risk.apply(lambda x: clean_display_value(x, "N/A")) == sr]
            if sc != "Tümü": filtered = filtered[filtered[cat_col].apply(lambda x: clean_display_value(x, "N/A")) == sc]
            if sf != "Tümü": filtered = filtered[filtered.rf_display_label.apply(lambda x: clean_display_value(x, "N/A")) == sf]
            if ss != "Tümü": filtered = filtered[filtered.source_file.astype(str) == ss]
            if only_mismatch: filtered = filtered[filtered.model_mismatch == True]
            show_cols = ["source_file","source_row","timestamp","true_label","true_display_label","true_category","risk","rf_display_label","rf_category","rf_confidence","rf_true_label_match","binary_status","if_score","if_anom","ae_error","ae_anom","feature_behavior","ifae_risk","rf_override","decision_reason","decision_detail","action","model_mismatch"]
            show_cols = [c for c in show_cols if c in filtered.columns]
            display_limit = 500
            show = filtered[show_cols].head(display_limit).copy()
            show.columns = ["Dosya","Kaynak Satır","Zaman","Gerçek Label","Okunabilir Tür","Kategori","Risk","RF Tahmini","RF Kategorisi","RF Güven (%)","RF-Label Eşleşti mi?","Binary Durum","IF Skoru","IF Anomali","AE Hatası","AE Anomali","Feature Davranış Özeti","IF/AE Başlangıç Risk","RF Yükseltme","Karar Gerekçesi","Karar Detayı","Önerilen Aksiyon","Model Uyuşmazlığı"][:len(show.columns)]
            st.dataframe(show.style.map(color_risk, subset=["Risk"]), width="stretch", hide_index=True)
            if len(filtered) > display_limit:
                st.caption(f"Performans için tabloda ilk {display_limit} kayıt gösteriliyor. CSV çıktısı analiz edilen tüm kayıtları içerir. Filtre sonrası toplam kayıt: {len(filtered)}")
            else:
                st.caption(f"{len(filtered)} kayıt görüntüleniyor.")
            st.divider(); st.subheader("📤 Rapor ve Çıktılar")
            d1,d2 = st.columns(2)
            with d1: st.download_button("⬇️ Sonuçları İndir (CSV)", results_df.to_csv(index=False).encode("utf-8-sig"), "analiz_sonuclari_gelismis.csv", "text/csv", width="stretch")
            with d2:
                if REPORTLAB_AVAILABLE:
                    if st.button("📄 PDF Raporu Hazırla", width="stretch"):
                        with st.spinner("PDF raporu hazırlanıyor..."):
                            st.session_state.pdf_report_buffer = create_pdf_report(results_df, binary_metrics)
                    if st.session_state.get("pdf_report_buffer") is not None:
                        st.download_button("⬇️ Hazırlanan PDF Raporu İndir", st.session_state.pdf_report_buffer, "ids_analiz_raporu.pdf", "application/pdf", width="stretch")
                else:
                    st.warning("PDF için reportlab gerekli: pip install reportlab")

# ──────────────────────────────────────────────────────────────
# Tab 3
# ──────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Sistem Mimarisi ve Çalışma Mantığı")
    st.markdown('<div class="info-box">Bu sekme, uygulamanın model kararlarını nasıl ürettiğini ve sonuçların nasıl yorumlanması gerektiğini özetler.</div>', unsafe_allow_html=True)

    st.markdown("### Karar Akış Diyagramı")
    render_decision_flow_diagram()
    st.caption("Nihai risk kararı RF destekli hibrit karar motoru ile üretilir. Isolation Forest ve Autoencoder davranışsal anomali sinyali sağlar; Random Forest yüksek güvenli saldırı tahminlerinde LOW kararını MEDIUM inceleme seviyesine yükseltebilir.")
    st.divider()

    with st.expander("1. Sistem Özeti", expanded=True):
        st.markdown("""
        Bu uygulama, CICIDS2017 formatındaki ağ akışı özelliklerini kullanarak anomali tespiti ve saldırı sınıfı tahmini yapan makine öğrenmesi tabanlı bir IDS prototipidir. Sistem; **Isolation Forest**, **Autoencoder** ve **Random Forest** modellerini birlikte kullanır. Ancak bu üç model aynı işi yapmaz: IF ve AE davranışsal anomali sinyali üretir; RF ise saldırı sınıfı tahmini ve yüksek güvenli durumlarda risk yükseltme desteği sağlar.
        """)

    with st.expander("2. Kullanılan Modeller ve Görevleri", expanded=True):
        model_df = pd.DataFrame({
            "Model":["Isolation Forest","Autoencoder","Random Forest"],
            "Tür":["Unsupervised","Unsupervised / Deep Learning","Supervised"],
            "Sistemdeki Rolü":["Normalden sapan ağ akışlarını tespit eder.","Normal trafiği öğrenir; yüksek reconstruction error üreten kayıtları anomali olarak değerlendirir.","Trafiğin hangi saldırı sınıfına benzediğini tahmin eder."],
            "Nihai Risk Kararına Etkisi":["Anomali sinyali olarak doğrudan katılır","Anomali sinyali olarak doğrudan katılır","Yüksek güvenli saldırı tahmininde destekli yükseltme yapar"]
        })
        st.dataframe(model_df, width="stretch", hide_index=True)

    with st.expander("3. Model Performansı", expanded=False):
        perf = pd.DataFrame(MODEL_PERFORMANCE_DATA)
        for c in ["Accuracy","Precision","Recall","F1"]:
            perf[c] = perf[c].apply(lambda x: f"{x:.4f}")
        st.dataframe(perf, width="stretch", hide_index=True)
        st.markdown('<div class="info-box"><b>Genel değerlendirme:</b> Random Forest sınıf tahmini ve saldırı yakalama oranı açısından güçlüdür. Autoencoder daha seçici davranır ve precision değeri yüksektir. Isolation Forest etiket kullanmadan genel anomali davranışını yakalamaya çalışır. Bu nedenle modeller tek bir accuracy değeriyle değil, görevlerine göre ayrı değerlendirilmelidir.</div>', unsafe_allow_html=True)

    with st.expander("4. Karar Motoru ve Algoritma Ağırlıkları", expanded=True):
        st.markdown("""
        Risk kararında **Isolation Forest** ve **Autoencoder** eşit ağırlıklı iki anomali kaynağı olarak kullanılır. Her modelin anomali kararı 1 oy olarak değerlendirilir. **Random Forest**, sınıf tahmini ve güven değeri üretir; eğer IF/AE başlangıç kararı LOW kalmasına rağmen RF yüksek güvenle saldırı sınıfı tahmin ederse kayıt MEDIUM seviyesine yükseltilir. Böylece saldırıyı kaçırma (false negative) riski azaltılmaya çalışılır.
        """)
        role_df = pd.DataFrame({
            "Model":["Isolation Forest","Autoencoder","Random Forest"],
            "Karardaki Rolü":["Anomali sinyali üretir","Anomali sinyali üretir","Sınıf tahmini ve destekli risk yükseltme üretir"],
            "Ağırlık / Kullanım":["IF anomali=True ise 1 oy","AE anomali=True ise 1 oy","RF güveni eşik üzerindeyse LOW → MEDIUM yükseltme"]
        })
        st.dataframe(role_df, width="stretch", hide_index=True)
        risk_df = pd.DataFrame({
            "Risk Seviyesi":["HIGH","MEDIUM","LOW"],
            "Koşul":["IF ve AE birlikte anomali derse veya tek anomaliye yüksek güvenli RF saldırı tahmini eşlik ederse","IF veya AE modellerinden yalnızca biri anomali derse ya da IF/AE LOW iken RF yüksek güvenle saldırı tahmin ederse","IF/AE normal ve RF destekli yükseltme yoksa"],
            "Önerilen Aksiyon":["Trafiği engelle ve olayı incelemeye al.","Analist incelemesine yönlendir.","Trafiğe izin ver."]
        })
        st.dataframe(risk_df, width="stretch", hide_index=True)

    with st.expander("5. Top Feature Listesi", expanded=False):
        st.dataframe(pd.DataFrame({"Sıra": range(1, len(top_features)+1), "Feature": top_features}), width="stretch", hide_index=True)
        st.caption("Bu liste eğitim sürecinde seçilen ve uygulama tarafında beklenen feature kolonlarını gösterir. Label kolonu model girişine dahil edilmez.")

    render_terms_glossary(expanded=False)

    with st.expander("6. Label Mapping ve Saldırı Kategorileri", expanded=False):
        mapping_df = pd.DataFrame([{"Normalize Edilmiş Label":k,"Ekranda Gösterilen İsim":v[0],"Kategori":v[1]} for k,v in LABEL_MAP.items()])
        st.dataframe(mapping_df, width="stretch", hide_index=True)

    with st.expander("7. Analiz Modları", expanded=False):
        mode_df = pd.DataFrame({
            "Analiz Modu":[
                "İlk N satır",
                "Rastgele N satır",
                "Gerçek dağılıma yakın örneklem",
                "Binary dengeli örneklem",
                "Label dengeli örneklem",
                "Sadece saldırı kayıtları",
                "Seçili label"
            ],
            "Açıklama":[
                "CSV dosyasının başından seçilen satır sayısı kadar analiz yapar.",
                "CSV dosyasından rastgele örneklem alır.",
                "Veri setindeki doğal dağılıma yakın rastgele örneklem üretir.",
                "BENIGN ve ATTACK kayıtlarını yaklaşık dengeli seçer.",
                "Her orijinal label sınıfından mümkün olduğunca dengeli örnek seçer; saldırı sınıflarını daha görünür yapabilir.",
                "BENIGN dışındaki kayıtları analiz eder.",
                "Kullanıcının seçtiği belirli label üzerinde analiz yapar."
            ]
        })
        st.dataframe(mode_df, width="stretch", hide_index=True)

    with st.expander("8. Outlier ve t-SNE Kullanımı", expanded=False):
        st.markdown("""
        **Outlier**, normal ağ trafiği davranışından belirgin şekilde ayrılan akışları ifade eder. Ağ güvenliği veri setlerinde outlier kayıtlar her zaman silinmesi gereken hatalı kayıtlar değildir; bazı DoS, PortScan veya exploit davranışları doğal olarak uç değerler üretebilir.

        **t-SNE**, veri dağılımını iki boyutta görselleştirmek için kullanılmıştır. t-SNE bir sınıflandırma modeli değildir; veri içindeki benzerlikleri ve sınıfların birbirinden ayrışma eğilimini görsel olarak incelemek için kullanılır.
        """)

    with st.expander("9. Hata Yönetimi ve Çıktı Destekleri", expanded=False):
        cap_df = pd.DataFrame({
            "Özellik":["Model dosyası kontrolü","Çoklu CSV yükleme","Label mapping","Gerçek label bazlı performans","Model uyuşmazlığı analizi","Top 10 şüpheli akış","CSV export","PDF rapor"],
            "Durum":["Aktif","Aktif","Aktif","Label varsa aktif","Aktif","Aktif","Aktif","reportlab kuruluysa aktif"]
        })
        st.dataframe(cap_df, width="stretch", hide_index=True)

    st.divider()
    st.markdown('<div class="warning-box"><b>Limitasyon:</b><br>Bu uygulama akademik/prototip amaçlıdır. Gerçek canlı ağ trafiğini doğrudan yakalamaz. Analiz, CICIDS2017 formatındaki CSV dosyaları veya manuel girilen feature değerleri üzerinden yapılır. Gerçek ortam kullanımı için canlı trafik yakalama, CICFlowMeter entegrasyonu, eşik optimizasyonu ve model yeniden eğitimi gibi ek geliştirmeler gerekir.</div>', unsafe_allow_html=True)
