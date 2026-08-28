import streamlit as st
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import numpy as np

# 1. Корпоративна конфигурация на интерфейса
st.set_page_config(
    page_title="Industrial QC Matrix", 
    page_icon="🛡️", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Луксозен дизайн с големи и бели букви
st.markdown("""
    <style>
    @import url('https://googleapis.com');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: #0d1117;
        color: #e1e7ed;
        font-size: 18px;
    }
    h1 {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 800 !important;
        font-size: 46px !important;
        letter-spacing: -1px;
    }
    h2 {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 700 !important;
        font-size: 32px !important;
    }
    h3 {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 700 !important;
        font-size: 26px !important;
        letter-spacing: -0.5px;
    }
    
    /* СТИЛ ЗА КУТИЯТА/ПАНЕЛА ЗА КАЧВАНЕ */
    [data-testid="stFileUploaderDropzone"] {
        border: 2px dashed #30363d !important;
        border-radius: 12px !important;
        background-color: #161b22 !important;
        padding: 30px 20px !important;
        text-align: center !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    /* ЗЕЛЕНИЯТ КОРПОРАТИВЕН БУТОН */
    [data-testid="stFileUploaderDropzone"] button {
        background: linear-gradient(135deg, #238636 0%, #2ea043 100%) !important;
        color: white !important;
        border: none !important;
        padding: 12px 28px !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-size: 18px !important;
        box-shadow: 0 4px 12px rgba(46, 160, 67, 0.2) !important;
        margin: 0 auto 15px auto !important;
        display: block !important;
    }
    [data-testid="stFileUploaderDropzone"] button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 15px rgba(46, 160, 67, 0.4) !important;
    }
    
    /* ТЕХНИЧЕСКИ ТЕКСТ ПОД БУТОНА ВЪТРЕ В СИВИЯ ПАНЕЛ */
    [data-testid="stFileUploaderDropzone"]::after {
        content: "📊 Поддържани формати: JPG, JPEG, PNG  |  📦 Макс. размер: 200 MB";
        display: block !important;
        color: #ffffff !important; 
        font-size: 16px !important; 
        font-weight: 700 !important; 
        margin-top: 5px !important;
        text-align: center !important;
        width: 100% !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.5); 
    }
    
    [data-testid="stFileUploaderDropzone"] section {
        display: none !important;
    }
    
    .metric-card {
        background: rgba(22, 27, 34, 0.7);
        border: 1px solid rgba(48, 54, 61, 0.8);
        border-radius: 12px;
        padding: 28px;
        backdrop-filter: blur(8px);
        box-shadow: 0 6px 24px rgba(0,0,0,0.3);
    }
    .metric-card p {
        font-size: 18px !important;
        line-height: 1.6;
    }
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 48px;
        font-weight: 700;
        margin-top: 10px;
    }
    .probability-text {
        font-size: 17px !important;
        font-weight: 600;
        color: #f0f6fc;
    }
    
    .custom-info-box {
        background: rgba(56, 139, 253, 0.15);
        border: 1px solid rgba(56, 139, 253, 0.4);
        border-radius: 8px;
        padding: 20px;
        color: #ffffff !important; 
        font-size: 18px;
        font-weight: 700; 
        line-height: 1.5;
        margin-top: 10px;
        box-shadow: 0 4px 12px rgba(56, 139, 253, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# 2. ЧИСТО ОБЛАЧНО ЗАРЕЖДАНЕ (Задължително за хостинг)
@st.cache_resource
def load_industrial_vision_core():
    # Облакът сам сваля модела в RAM паметта си при първия старт
    return SentenceTransformer('clip-ViT-B-32')

try:
    model = load_industrial_vision_core()
except Exception as e:
    st.error(f"Системна грешка при инициализация: {e}")
    st.stop()

# 3. Странично меню (Sidebar)
with st.sidebar:
    st.markdown("<h2 style='color:#58a6ff; margin-top:0;'>⚙️ QC Настройки</h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:16px;'>Конфигуриране на сензорите и праговете на толеранс в реално време.</p>", unsafe_allow_html=True)
    
    tolerance_threshold = st.slider("Критичен праг на сигурност (%)", min_value=50, max_value=95, value=75, step=5)
    
    st.write("---")
    st.markdown("<b style='color:#8b949e; font-size:16px;'>Системна информация:</b>", unsafe_allow_html=True)
    st.caption("Engine: OpenAI CLIP ViT-B-32")
    st.caption("Hardware: Cloud CPU Instance")
    st.caption("Mode: Automated Quality Control")

# 4. Основен екран / Дашборд
st.markdown("<h1 style='color:#f0f6fc; margin-bottom: 0px;'>🛡️ MATRIX QUALITY CONTROL</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#8b949e; font-size:18px; margin-top:5px; margin-bottom:30px;'>Автоматизиран визуален контрол и откриване на производствени дефекти.</p>", unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1.2], gap="large")

with col_left:
    st.markdown("<h3 style='color:#58a6ff; margin-bottom:15px;'>📥 Входящ визуален поток</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Качване на файл", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file is not None:
        st.write("")
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Входяща матрица от камерата', width='stretch')

with col_right:
    st.markdown("<h3 style='color:#58a6ff; margin-bottom:15px;'>📊 Системен анализ на детайла</h3>", unsafe_allow_html=True)
    
    if uploaded_file is not None:
        with st.spinner("Математическа екстракция на сигнатура..."):
            image_embedding = model.encode(image, convert_to_tensor=True)
        
        class_queries = [
            "a green or red apple with a stem and leaf, fresh round fruit",
            "a bunch of bananas, yellow or red banana, elongated fruit",
            "a round orange citrus fruit, texturized skin orange"
        ]
        
        text_embeddings = model.encode(class_queries, convert_to_tensor=True)
        cos_scores = util.cos_sim(image_embedding, text_embeddings)
        scores = cos_scores.cpu().numpy()
        
        exp_scores = np.exp(scores * 50)  
        probabilities = (exp_scores / np.sum(exp_scores))
        
        top_class_idx = np.argmax(probabilities)
        confidence = float(probabilities[top_class_idx] * 100)
        
        system_labels = ["Компонент А (Ябълка)", "Компонент B (Банан)", "Компонент C (Портокал)"]
        detected_label = system_labels[top_class_idx]
        
        if confidence < tolerance_threshold:
            st.markdown(f"""
                <div class='metric-card' style='border-left: 6px solid #f85149;'>
                    <h3 style='color:#f85149; margin:0;'>🚨 КРИТИЧНА АНОМАЛИЯ</h3>
                    <p style='color:#8b949e; margin:6px 0 16px 0;'>Обектът се отклонява от геометричните стандарти или съдържа дефект.</p>
                    <div style='color:#f85149; font-size:20px;'>Най-близка структура: <b>{detected_label}</b></div>
                    <div class='metric-value' style='color:#f85149;'>{confidence:.2f}%</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class='metric-card' style='border-left: 6px solid #56d364;'>
                    <h3 style='color:#56d364; margin:0;'>✅ КОМПОНЕНТЪТ Е ОДОБРЕН</h3>
                    <p style='color:#8b949e; margin:6px 0 16px 0;'>Пълно съответствие с производствените спецификации.</p>
                    <div style='color:#56d364; font-size:20px;'>Идентифициран клас: <b>{detected_label}</b></div>
                    <div class='metric-value' style='color:#56d364;'>{confidence:.2f}%</div>
                </div>
            """, unsafe_allow_html=True)
            
        st.write("")
        st.markdown("<b style='color:#f0f6fc; font-size:19px;'>Матрица на вероятностите:</b>", unsafe_allow_html=True)
        for idx, label in enumerate(system_labels):
            st.markdown(f"<div class='probability-text' style='margin-top:10px;'>{label} ({probabilities[idx]*100:.1f}%)</div>", unsafe_allow_html=True)
            st.progress(float(probabilities[idx]))
    else:
        st.markdown("""
            <div class='custom-info-box'>
                ℹ️ Очакване на входящ визуален сигнал от камерата за стартиране на системната инспекция.
            </div>
        """, unsafe_allow_html=True)
