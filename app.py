import streamlit as st
import pickle
import numpy as np
from TurkishStemmer import TurkishStemmer
import re
import nltk
import os
import base64
import sqlite3
from datetime import datetime
import sklearn.tree._tree as _tree
import warnings
import sklearn
from sklearn.base import BaseEstimator
import joblib
import time
from tensorflow import keras

# NLTK stopwords'ü yükle
nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('turkish')
porter = TurkishStemmer()

# Başa eklenecek importlar ve fonksiyonlar
def format_time_ago(created_date_str):
    """Geçen süreyi hesapla ve formatla"""
    try:
        created_date = datetime.strptime(created_date_str, '%Y-%m-%d %H:%M:%S')
        now = datetime.now()
        diff = now - created_date
        
        if diff.days > 365:
            years = diff.days // 365
            return f"{years} yıl önce"
        if diff.days > 30:
            months = diff.days // 30
            return f"{months} ay önce"
        if diff.days > 0:
            return f"{diff.days} gün önce"
        if diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours} saat önce"
        if diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes} dakika önce"
        return f"{diff.seconds} saniye önce"
    except Exception as e:
        print(f"Tarih formatı hatası: {e}")
        return "Tarih bilgisi alınamadı"

# Veritabanı fonksiyonları
def init_db():
    conn = sqlite3.connect('complaints.db')
    c = conn.cursor()
    
    # Kullanıcılar tablosu
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            phone TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Şikayetler tablosu (user_id eklendi)
    c.execute('''
        CREATE TABLE IF NOT EXISTS complaints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            complaint_number INTEGER NOT NULL,
            complaint_text TEXT NOT NULL,
            category TEXT NOT NULL,
            model_used TEXT NOT NULL,
            status TEXT DEFAULT 'İnceleniyor',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Admin tablosunu ekle
    c.execute('''
        CREATE TABLE IF NOT EXISTS admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # İlk admin kullanıcısını ekle (eğer yoksa)
    c.execute('SELECT COUNT(*) FROM admins')
    if c.fetchone()[0] == 0:
        # Şifre: admin123 (gerçek uygulamada hash'lenmiş olmalı)
        c.execute('INSERT INTO admins (username, password) VALUES (?, ?)',
                 ('admin', 'admin123'))
    
    conn.commit()
    conn.close()

def save_user(name, email, phone):
    conn = sqlite3.connect('complaints.db')
    c = conn.cursor()
    try:
        c.execute('INSERT INTO users (name, email, phone) VALUES (?, ?, ?)',
                 (name, email, phone))
        conn.commit()
        user_id = c.lastrowid
        return user_id
    except sqlite3.IntegrityError:
        # Email zaten varsa, mevcut user_id'yi döndür
        c.execute('SELECT id FROM users WHERE email = ?', (email,))
        user_id = c.fetchone()[0]
        return user_id
    finally:
        conn.close()

def get_user_complaints(email):
    conn = sqlite3.connect('complaints.db')
    c = conn.cursor()
    try:
        c.execute('SELECT id FROM users WHERE email = ?', (email,))
        user = c.fetchone()
        
        if not user:
            return []
            
        c.execute('''
            SELECT c.complaint_number, c.complaint_text, c.category, 
                   c.status, c.created_at, u.name, u.email, c.model_used
            FROM complaints c
            JOIN users u ON c.user_id = u.id
            WHERE u.email = ?
            ORDER BY c.created_at DESC
        ''', (email,))
        
        return c.fetchall()
        
    except Exception as e:
        print(f"Şikayet sorgulama hatası: {e}")
        return []
    finally:
        conn.close()

def save_complaint(user_id, complaint_number, complaint_text, category, model_used):
    conn = sqlite3.connect('complaints.db')
    c = conn.cursor()
    try:
        # Model adını düzgün formata çevir
        model_mapping = {
            'mb': 'MultinomialNB',
            'sgd': 'SGD Classifier',
            'lr': 'Logistic Regression'
        }
        formatted_model = model_mapping.get(model_used, model_used)
        
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        c.execute('''
            INSERT INTO complaints 
            (user_id, complaint_number, complaint_text, category, model_used, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, complaint_number, complaint_text, category, formatted_model, 'İnceleniyor', current_time))
        conn.commit()
        print(f"Şikayet kaydedildi: {complaint_number}, Model: {formatted_model}")  # Debug için
    except Exception as e:
        print(f"Şikayet kaydetme hatası: {e}")  # Debug için
        conn.rollback()
    finally:
        conn.close()

def get_complaint_by_number(complaint_number):
    conn = sqlite3.connect('complaints.db')
    c = conn.cursor()
    try:
        c.execute('''
            SELECT c.complaint_number, c.complaint_text, c.category, 
                   c.status, c.created_at, u.name, u.email
            FROM complaints c
            JOIN users u ON c.user_id = u.id
            WHERE c.complaint_number = ?
        ''', (int(complaint_number),))
        
        return c.fetchone()
        
    except Exception as e:
        return None
    finally:
        conn.close()

def verify_admin(username, password):
    conn = sqlite3.connect('complaints.db')
    c = conn.cursor()
    c.execute('SELECT id FROM admins WHERE username = ? AND password = ?',
             (username, password))
    result = c.fetchone()
    conn.close()
    return result is not None

def update_complaint_status(complaint_number, new_status):
    conn = sqlite3.connect('complaints.db')
    c = conn.cursor()
    c.execute('UPDATE complaints SET status = ? WHERE complaint_number = ?',
             (new_status, complaint_number))
    conn.commit()
    conn.close()

def get_all_complaints():
    conn = sqlite3.connect('complaints.db')
    c = conn.cursor()
    try:
        c.execute('''
            SELECT 
                c.complaint_number,
                c.complaint_text,
                c.category,
                c.status,
                c.created_at,
                u.name,
                u.email,
                u.phone,
                c.model_used  -- model_used'ı da seç
            FROM complaints c
            JOIN users u ON c.user_id = u.id
            ORDER BY c.created_at DESC
        ''')
        complaints = c.fetchall()
        print(f"Toplam {len(complaints)} şikayet bulundu")  # Debug için
        return complaints
    except Exception as e:
        print(f"Şikayet getirme hatası: {e}")  # Debug için
        return []
    finally:
        conn.close()

# Veritabanını başlat
init_db()

# Modelleri ve vectorizer'ı yükle
print(f"Scikit-learn version: {sklearn.__version__}")

try:
    warnings.filterwarnings('ignore')
    
    # Model yolunu düzelt
    model_path = "models"
    
    # Mevcut modelleri yükle
    with open(os.path.join(model_path, 'multinomial_nb_model.pkl'), 'rb') as f:
        mb_model = pickle.load(f)
    
    with open(os.path.join(model_path, 'count_vectorizer.pkl'), 'rb') as f:
        count_vectorizer = pickle.load(f)
    
    with open(os.path.join(model_path, 'logistic_model.pkl'), 'rb') as f:
        best_lr_model = pickle.load(f)
        
    with open(os.path.join(model_path, 'sgd_model.pkl'), 'rb') as f:
        sgd_model = pickle.load(f)

    # Deep Learning modelini yükle (custom_objects ile)
    try:
        dl_model = keras.models.load_model(
            os.path.join(model_path, 'deep_learning_model.keras'),
            custom_objects=None,  # Eğer özel katmanlar/metrikler varsa burada belirtin
            compile=False  # Modeli compile etmeden yükle
        )
        dl_model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    except Exception as dl_error:
        print(f"Deep Learning model yükleme hatası: {dl_error}")
        dl_model = None

    # Label Binarizer'ı yükle
    with open(os.path.join(model_path, 'label_binarizer.pkl'), 'rb') as f:
        label_binarizer = pickle.load(f)

    # Label mapping'i yükle
    with open(os.path.join(model_path, 'label_mapping.pkl'), 'rb') as f:
        label_mapping = pickle.load(f)

except FileNotFoundError as e:
    st.error(f"Model dosyası bulunamadı: {str(e)}")
    st.error(f"Aranan yol: {model_path}")
    st.error(f"Dizindeki dosyalar: {os.listdir(model_path) if os.path.exists(model_path) else 'Dizin bulunamadı'}")
    st.stop()
except Exception as e:
    st.error(f"Model yükleme hatası: {str(e)}")
    st.stop()

def find_label(sentence, model='mb'):
    """Şikayet kategorisini belirle"""
    try:
        cleaned_text = clean_text(sentence)
        text_vector = count_vectorizer.transform([cleaned_text])
        
        if model == 'dl':  # Deep Learning için yeni seçenek
            if dl_model is None:
                st.warning("Deep Learning modeli yüklenemedi. Varsayılan model (MultinomialNB) kullanılıyor.")
                prediction = mb_model.predict(text_vector)[0]
            else:
                # Text vector'ü dense array'e çevir
                text_array = text_vector.toarray()
                # Tahmin yap
                prediction_proba = dl_model.predict(text_array)
                prediction = label_binarizer.inverse_transform(prediction_proba > 0.5)[0]
        elif model == 'sgd':
            prediction = sgd_model.predict(text_vector)[0]
        elif model == 'lr':
            prediction = best_lr_model.predict(text_vector)[0]
        else:  # mb (MultinomialNB)
            prediction = mb_model.predict(text_vector)[0]
        
        labels = {
            0: 'İgdaş (Doğalgaz dağıtımı ve faturalandırma)',
            1: 'İett (Toplu taşıma)',
            2: 'İski (Su dağıtımı ve faturalandırma)',
            3: 'Diğer İBB iştirakleri',
            4: 'İlgisiz'
        }
        
        return labels[prediction]
        
    except Exception as e:
        st.error(f"Seçilen model ({model}) ile tahmin yapılamadı: {str(e)}")
        return None

def clean_text(string):
    """Metin temizleme fonksiyonu"""
    message = re.sub(r'\b[\w\-.]+?@\w+?\.\w{2,4}\b', 'emailaddr', str(string))
    message = re.sub(r'(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)', ' ', message)
    message = re.sub(r'₺|\$', 'money', message)
    message = re.sub(r'\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b', 'phonenumbr', message)
    message = re.sub(r'\d+(\.\d+)?', 'numbr', message)
    message = re.sub(r'[^\w\d\s]', ' ', message)
    message = re.sub(r'\s+', ' ', message)
    message = re.sub(r'^\s+|\s+?$', '', message.lower())
    return ' '.join(
        porter.stem(term)
        for term in message.split()
        if term not in set(stop_words)
    )

# CSS stillerini güncelle
st.markdown("""
<style>
    /* Ana tema renkleri */
    :root {
        --primary-color: #0083b0;
        --secondary-color: #00b4db;
        --background-color: #f8f9fa;
        --text-color: #2c3e50;
    }

    /* Animasyonlar için keyframe tanımlamaları */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }

    @keyframes slideIn {
        from { transform: translateX(-100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }

    /* Header container stili */
    .header-container {
        background: linear-gradient(135deg, #2c3e50, #3498db);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        animation: slideIn 0.8s ease-out;
    }

    /* Buton stili */
    .stButton>button {
        width: 100%;
        border-radius: 25px !important;
        height: 3em;
        background: linear-gradient(45deg, var(--secondary-color), var(--primary-color)) !important;
        color: white !important;
        font-weight: bold !important;
        border: none !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
        position: relative;
        overflow: hidden;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0,0,0,0.3) !important;
        animation: pulse 1s infinite;
    }
    
    .stButton>button:active {
        transform: translateY(1px) !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2) !important;
    }

    /* Metin alanı stili */
    .css-1v0mbdj.ebxwdo61, .st-emotion-cache-1v0mbdj.ebxwdo61 {
        border-radius: 15px;
        border: 2px solid rgba(0,131,176,0.1);
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        background-color: white;
        transition: all 0.3s ease;
        animation: fadeIn 0.5s ease-out;
    }
    
    .css-1v0mbdj.ebxwdo61:focus-within {
        border-color: var(--primary-color);
        box-shadow: 0 4px 20px rgba(0,131,176,0.15);
        transform: translateY(-2px);
    }

    /* Şikayet kutusu stili */
    .complaint-box {
        padding: 1.5rem;
        border-radius: 15px;
        background: rgba(255, 255, 255, 0.1);
        margin: 1rem 0;
        border-left: 5px solid var(--primary-color);
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        animation: fadeIn 0.5s ease-out;
    }
    
    .complaint-box:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        background: rgba(255, 255, 255, 0.15);
    }

    /* Radio butonları stili */
    .st-emotion-cache-1v0mbdj {
        background-color: white;
        border-radius: 10px;
        padding: 0.5rem;
        transition: all 0.3s ease;
    }
    
    .st-emotion-cache-1v0mbdj:hover {
        background-color: rgba(255,255,255,0.9);
        transform: translateY(-1px);
    }

    /* Success message stili */
    .st-emotion-cache-1eqh5xj {
        border-radius: 10px;
        animation: fadeIn 0.5s ease-out;
        transition: all 0.3s ease;
    }

    /* Loading spinner stili */
    .stSpinner {
        animation: pulse 1s infinite;
    }

    /* Hover efektleri */
    .complaint-box p strong {
        color: var(--primary-color);
        transition: all 0.3s ease;
    }
    
    .complaint-box:hover p strong {
        color: var(--secondary-color);
    }

    /* Tab stili */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        border-radius: 10px;
        background-color: rgba(255,255,255,0.1);
        transition: all 0.3s ease;
        border: none !important;
        padding: 0 20px;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(255,255,255,0.2);
        transform: translateY(-2px);
    }

    .stTabs [aria-selected="true"] {
        background-color: var(--primary-color) !important;
        color: white !important;
    }

    /* Genel animasyonlar */
    .stMarkdown, .stText {
        animation: fadeIn 0.5s ease-out;
    }

    /* Şikayet kartları grid container */
    .complaints-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 20px;
        padding: 20px 0;
    }

    /* Şikayet kartı stili */
    .complaint-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        border-left: 5px solid var(--primary-color);
        margin-bottom: 20px;
        animation: fadeIn 0.5s ease-out;
    }
    
    .complaint-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
        background: linear-gradient(145deg, rgba(255, 255, 255, 0.15), rgba(255, 255, 255, 0.1));
    }

    @keyframes cardPop {
        0% {
            opacity: 0;
            transform: scale(0.95) translateY(20px);
        }
        100% {
            opacity: 1;
            transform: scale(1) translateY(0);
        }
    }

    /* Status badge stili */
    .status-badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 0.8em;
        font-weight: bold;
        background: var(--primary-color);
        color: white;
        margin-top: 10px;
    }

    /* Şikayet içeriği stili */
    .complaint-content {
        flex-grow: 1;
        margin: 15px 0;
    }

    .complaint-content p {
        margin: 5px 0;
    }

    /* Footer stili */
    .complaint-footer {
        margin-top: 15px;
        padding-top: 15px;
        border-top: 1px solid rgba(0, 0, 0, 0.1);
    }

    /* Zaman bilgisi stili */
    .time-info {
        display: flex;
        flex-direction: column;
        gap: 8px;
        font-size: 0.9em;
    }

    .created-date, .time-ago {
        display: flex;
        align-items: center;
        padding: 6px 12px;
        background: rgba(255, 255, 255, 0.15);
        border-radius: 8px;
        color: #2c3e50;
        font-weight: 500;
        transition: all 0.3s ease;
        backdrop-filter: blur(5px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    .created-date:hover, .time-ago:hover {
        background: rgba(255, 255, 255, 0.25);
        transform: translateX(5px);
    }

    .icon {
        width: 16px;
        height: 16px;
        margin-right: 8px;
        color: #2c3e50;
        opacity: 0.8;
    }

    .time-info span:hover .icon {
        opacity: 1;
    }

    /* Grid düzeni için ek stiller */
    .st-emotion-cache-12w0qpk {
        gap: 1rem;
    }

    .st-emotion-cache-1r6slb0 {
        padding: 0 0.5rem;
    }

    .icon {
        vertical-align: middle;
        margin-right: 5px;
        opacity: 0.8;
    }

    .time-info span:hover .icon {
        opacity: 1;
        transform: scale(1.1);
    }

    .time-info span {
        display: flex;
        align-items: center;
        gap: 5px;
    }

    /* Model bazlı şikayetler için ek stiller */
    .model-section {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 15px;
        margin: 20px 0;
        border-left: 5px solid var(--primary-color);
    }

    .model-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 15px;
    }

    .model-count {
        background: var(--primary-color);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# CSS'e Font Awesome CDN'ini ekleyelim
st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
""", unsafe_allow_html=True)

# Logo dosyasını base64'e çevir
def get_base64_from_file(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Logo'yu base64'e çevir
logo_base64 = get_base64_from_file("/Users/kemalsongur/Desktop/Masaüstü - Kemal's MacBook Pro/YZO/ibb-logo.svg")

# Header'ı güncelle
st.markdown("""
    <div class="header-container">
        <div style="display: grid; grid-template-columns: 250px 1fr; gap: 30px;">
            <div style="background: white; padding: 15px; border-radius: 15px; display: flex; justify-content: center; align-items: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                <img src="data:image/svg+xml;base64,{}" style="height: 150px; transition: all 0.3s ease;" onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'">
            </div>
            <div style="display: flex; flex-direction: column; justify-content: center;">
                <h1 style='margin: 0; font-size: 2.5em; background: linear-gradient(45deg, #ffffff, #f0f0f0); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                    İBB Şikayet Kategorilendirme Sistemi
                </h1>
                <p style='margin: 10px 0 0 0; font-size: 1.2em; color: rgba(255,255,255,0.9);'>
                    Yapay Zeka Destekli Şikayet Yönetim Sistemi
                </p>
            </div>
        </div>
    </div>
""".format(logo_base64), unsafe_allow_html=True)

# Tabs oluştur
tab1, tab2, tab3 = st.tabs(["📝 Yeni Şikayet", "🔍 Şikayet Sorgula", "👨‍💼 Admin Paneli"])

with tab1:
    st.markdown("### 👤 Kişisel Bilgiler")
    
    # Kişisel bilgiler formu
    with st.form("user_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Ad Soyad*")
            email = st.text_input("E-posta*")
        with col2:
            phone = st.text_input("Telefon")
        
        st.markdown("### 🤖 Model Seçimi")
        model_choice = st.selectbox(
            "Model Seçin",
            ["MultinomialNB", "SGD Classifier", "Logistic Regression", "Deep Learning"],
            key="model_select"
        )
        
        # Model mapping'i güncelle
        model_type_mapping = {
            "MultinomialNB": "mb",
            "SGD Classifier": "sgd",
            "Logistic Regression": "lr",
            "Deep Learning": "dl"
        }
        
        st.markdown("### ✍️ Şikayet Detayları")
        complaint_text = st.text_area(
            "Lütfen şikayetinizi detaylı bir şekilde açıklayın:*",
            height=120
        )
        
        submit_button = st.form_submit_button("🚀 Şikayet Oluştur")
        
        if submit_button:
            if not name or not email or not complaint_text:
                st.error("⚠️ Lütfen zorunlu alanları doldurun!")
            else:
                with st.spinner("🔄 Şikayetiniz yapay zeka tarafından analiz ediliyor..."):
                    try:
                        # Kullanıcıyı kaydet/güncelle
                        user_id = save_user(name, email, phone)
                        
                        # Model tipini al
                        model_type = model_type_mapping[model_choice]
                        
                        # Şikayeti analiz et
                        category = find_label(complaint_text, model_type)
                        
                        if category:
                            complaint_number = np.random.randint(100000, 999999)
                            
                            # Şikayeti kaydet
                            save_complaint(
                                user_id=user_id,
                                complaint_number=complaint_number,
                                complaint_text=complaint_text,
                                category=category,
                                model_used=model_choice  # Tam model adını kullan
                            )
                            
                            st.success("✅ Şikayetiniz başarıyla kaydedildi!")
                            st.write(f"Şikayet numaranız: {complaint_number}")
                            
                    except Exception as e:
                        st.error(f"Bir hata oluştu: {str(e)}")
                        print(f"Hata detayı: {str(e)}")  # Debug için

# Tab2 içinde, session state kullanarak e-posta adresini saklayalım
if 'search_email' not in st.session_state:
    st.session_state.search_email = ''

with tab2:
    st.markdown("### 🔍 Şikayet Sorgulama")
    
    search_method = st.radio(
        "Arama Yöntemi",
        ["E-posta ile Ara", "Şikayet Numarası ile Ara"]
    )
    
    if search_method == "E-posta ile Ara":
        search_email = st.text_input("E-posta Adresiniz", value=st.session_state.search_email)
        st.session_state.search_email = search_email
        
        show_complaints = st.button("Şikayetlerimi Göster")
        
        if show_complaints or st.session_state.search_email:
            if st.session_state.search_email:
                complaints = get_user_complaints(st.session_state.search_email.strip())
                if complaints:
                    st.success(f"📋 Toplam {len(complaints)} adet şikayet bulundu.")
                    
                    # Şikayetleri kategorilere göre grupla
                    categorized_complaints = {
                        'İgdaş': [],
                        'İett': [],
                        'İski': [],
                        'Diğer İBB': [],
                        'İlgisiz': []
                    }
                    
                    for complaint in complaints:
                        category = complaint[2]
                        if 'İgdaş' in category:
                            categorized_complaints['İgdaş'].append(complaint)
                        elif 'İett' in category:
                            categorized_complaints['İett'].append(complaint)
                        elif 'İski' in category:
                            categorized_complaints['İski'].append(complaint)
                        elif 'Diğer' in category:
                            categorized_complaints['Diğer İBB'].append(complaint)
                        else:
                            categorized_complaints['İlgisiz'].append(complaint)
                    
                    # Kategorilere göre sekmeleri oluştur
                    tabs = st.tabs([
                        f"🔥 İgdaş ({len(categorized_complaints['İgdaş'])})",
                        f"🚌 İett ({len(categorized_complaints['İett'])})",
                        f"💧 İski ({len(categorized_complaints['İski'])})",
                        f"🏢 Diğer İBB ({len(categorized_complaints['Diğer İBB'])})",
                        f"❓ İlgisiz ({len(categorized_complaints['İlgisiz'])})"
                    ])
                    
                    # Her sekme için şikayetleri göster
                    for tab_idx, (category, tab) in enumerate(zip(categorized_complaints.keys(), tabs)):
                        with tab:
                            if categorized_complaints[category]:
                                for complaint in categorized_complaints[category]:
                                    complaint_number = complaint[0]
                                    status = complaint[3]
                                    created_date = complaint[4]
                                    time_ago = format_time_ago(created_date)
                                    formatted_date = datetime.strptime(created_date, '%Y-%m-%d %H:%M:%S').strftime('%d.%m.%Y %H:%M')
                                    
                                    st.markdown(f"""
                                        <div class="complaint-card">
                                            <div class="complaint-header">
                                                <h4 style="margin:0; color: var(--primary-color);">Şikayet #{complaint_number}</h4>
                                                <div class="status-badge">{status}</div>
                                            </div>
                                            <div class="complaint-content">
                                                <p style="margin-top:10px;">
                                                    {complaint[1]}
                                                </p>
                                            </div>
                                            <div class="complaint-footer">
                                                <div class="time-info">
                                                    <span class="created-date">
                                                        <i class="far fa-calendar-alt icon"></i>
                                                        {formatted_date}
                                                    </span>
                                                    <span class="time-ago">
                                                        <i class="far fa-clock icon"></i>
                                                        {time_ago}
                                                    </span>
                                                </div>
                                            </div>
                                        </div>
                                        <br>
                                    """, unsafe_allow_html=True)
                            else:
                                st.info(f"Bu kategoride henüz şikayet bulunmuyor.")
                else:
                    st.warning("❌ Bu e-posta adresine ait şikayet bulunamadı.")
            else:
                st.error("⚠️ Lütfen e-posta adresinizi girin!")

    else:  # Şikayet Numarası ile Ara
        complaint_number = st.text_input("Şikayet Numarası", key="complaint_number")
        
        if st.button("Şikayeti Göster"):
            if complaint_number:
                try:
                    complaint = get_complaint_by_number(complaint_number)
                    if complaint:
                        st.success("📋 Şikayet bulundu.")
                        st.markdown(f"""
                            <div class="complaint-box">
                                <p><strong>Şikayet No:</strong> {complaint[0]}</p>
                                <p><strong>Şikayet Sahibi:</strong> {complaint[5]}</p>
                                <p><strong>E-posta:</strong> {complaint[6]}</p>
                                <p><strong>Şikayet:</strong> {complaint[1]}</p>
                                <p><strong>Kategori:</strong> {complaint[2]}</p>
                                <p><strong>Durum:</strong> {complaint[3]}</p>
                                <p><strong>Tarih:</strong> {complaint[4]}</p>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning("❌ Bu şikayet numarasına ait kayıt bulunamadı.")
                except ValueError:
                    st.error("⚠️ Lütfen geçerli bir şikayet numarası girin!")
            else:
                st.error("⚠️ Lütfen şikayet numarasını girin!")

# Yeni tab3 (Model Bazlı Şikayetler) ekle
with tab3:
    st.markdown("### 👨‍💼 Admin Girişi")
    
    if 'admin_logged_in' not in st.session_state:
        st.session_state.admin_logged_in = False
    
    if not st.session_state.admin_logged_in:
        with st.form("admin_login"):
            username = st.text_input("Kullanıcı Adı")
            password = st.text_input("Şifre", type="password")
            login_button = st.form_submit_button("Giriş Yap")
            
            if login_button:
                if verify_admin(username, password):
                    st.session_state.admin_logged_in = True
                    st.rerun()
                else:
                    st.error("Hatalı kullanıcı adı veya şifre!")
    else:
        # Çıkış yap butonu
        if st.button("Çıkış Yap", key="logout"):
            st.session_state.admin_logged_in = False
            st.rerun()
        
        # Admin sekmelerini oluştur
        admin_tab1, admin_tab2 = st.tabs(["📊 Tüm Şikayetler", "🤖 Model Bazlı Analiz"])
        
        with admin_tab1:
            st.markdown("### 📊 Tüm Şikayetler")
            
            # Şikayetleri getir
            complaints = get_all_complaints()
            
            if not complaints:
                st.warning("Henüz hiç şikayet bulunmuyor.")
            else:
                st.success(f"Toplam {len(complaints)} şikayet bulundu.")
                
                # Durum filtreleme
                status_filter = st.multiselect(
                    "Durum Filtrele",
                    ["İnceleniyor", "Çözümleniyor", "Çözüldü"],
                    default=["İnceleniyor", "Çözümleniyor", "Çözüldü"]
                )
                
                # Her şikayet için kart oluştur
                for complaint in complaints:
                    if complaint[3] in status_filter:  # status filtresi
                        with st.container():
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.markdown(f"""
                                    <div class="complaint-card">
                                        <h4>Şikayet #{complaint[0]}</h4>
                                        <p><strong>Müşteri:</strong> {complaint[5]}</p>
                                        <p><strong>E-posta:</strong> {complaint[6]}</p>
                                        <p><strong>Telefon:</strong> {complaint[7] or 'Belirtilmemiş'}</p>
                                        <p><strong>Kategori:</strong> {complaint[2]}</p>
                                        <p><strong>Şikayet:</strong> {complaint[1]}</p>
                                        <p><strong>Tarih:</strong> {complaint[4]}</p>
                                        <p><strong>Mevcut Durum:</strong> {complaint[3]}</p>
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                new_status = st.selectbox(
                                    "Durum Güncelle",
                                    ["İnceleniyor", "Çözümleniyor", "Çözüldü"],
                                    key=f"select_{complaint[0]}",
                                    index=["İnceleniyor", "Çözümleniyor", "Çözüldü"].index(complaint[3])
                                )
                                
                                if st.button("Güncelle", key=f"update_{complaint[0]}"):
                                    update_complaint_status(complaint[0], new_status)
                                    st.success("Durum güncellendi!")
                                    time.sleep(1)
                                    st.rerun()
        
        with admin_tab2:
            st.markdown("### 🤖 Model Bazlı Analiz")
            
            # Debug bilgisi göster
            st.markdown("#### Model Dağılımı")
            
            # Tüm şikayetleri modellere göre grupla
            model_complaints = {
                'MultinomialNB': [],
                'SGD Classifier': [],
                'Logistic Regression': [],
                'Deep Learning': []  # Deep Learning eklendi
            }
            
            # Her şikayeti uygun modele ekle
            for complaint in complaints:
                model = complaint[8] if complaint[8] else "Bilinmeyen"  # model_used 8. indekste
                if model in model_complaints:
                    model_complaints[model].append(complaint)
                elif model == "mb":
                    model_complaints['MultinomialNB'].append(complaint)
                elif model == "sgd":
                    model_complaints['SGD Classifier'].append(complaint)
                elif model == "lr":
                    model_complaints['Logistic Regression'].append(complaint)
                elif model == "dl":  # Deep Learning için yeni kontrol
                    model_complaints['Deep Learning'].append(complaint)
            
            # Model istatistiklerini göster
            col1, col2, col3, col4 = st.columns(4)  # 4 sütuna çıkarıldı
            
            with col1:
                nb_count = len(model_complaints['MultinomialNB'])
                st.metric("MultinomialNB", nb_count)
            with col2:
                sgd_count = len(model_complaints['SGD Classifier'])
                st.metric("SGD Classifier", sgd_count)
            with col3:
                lr_count = len(model_complaints['Logistic Regression'])
                st.metric("Logistic Regression", lr_count)
            with col4:  # Deep Learning için yeni metrik
                dl_count = len(model_complaints['Deep Learning'])
                st.metric("Deep Learning", dl_count)

            st.markdown("---")
            
            # Her model için ayrı bir bölüm oluştur
            for model_name, model_complaints_list in model_complaints.items():
                with st.expander(f"🔹 {model_name} ({len(model_complaints_list)} şikayet)"):
                    if model_complaints_list:
                        for complaint in model_complaints_list:
                            st.markdown(f"""
                                <div class="complaint-card">
                                    <div class="complaint-header">
                                        <h4>Şikayet #{complaint[0]}</h4>
                                        <div class="status-badge">{complaint[3]}</div>
                                    </div>
                                    <p><strong>Kategori:</strong> {complaint[2]}</p>
                                    <p><strong>Şikayet:</strong> {complaint[1]}</p>
                                    <p><strong>Tarih:</strong> {complaint[4]}</p>
                                </div>
                                <br>
                            """, unsafe_allow_html=True)
                    else:
                        st.info(f"Bu modele ait şikayet bulunmuyor.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>©  {year} İBB Şikayet Kategorilendirme Sistemi | Tüm Hakları Saklıdır</p>
        <p style='font-size: 0.9em;'>Geliştirici: <strong>Kemal Songur</strong></p>
    </div>
""".format(year=datetime.now().year), unsafe_allow_html=True)

# Veritabanını sıfırlamak için önce bu fonksiyonu ekleyelim
def reset_db():
    # Eğer veritabanı dosyası varsa sil
    if os.path.exists('complaints.db'):
        return  # Veritabanı varsa hiçbir şey yapma
    
    # Yeni veritabanını oluştur
    conn = sqlite3.connect('complaints.db')
    c = conn.cursor()
    
    # Kullanıcılar tablosu
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            phone TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Şikayetler tablosu
    c.execute('''
        CREATE TABLE IF NOT EXISTS complaints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            complaint_number INTEGER NOT NULL,
            complaint_text TEXT NOT NULL,
            category TEXT NOT NULL,
            model_used TEXT NOT NULL,
            status TEXT DEFAULT 'İnceleniyor',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()

# Veritabanını başlat (sıfırlama olmadan)
init_db()

def update_existing_statuses():
    conn = sqlite3.connect('complaints.db')
    c = conn.cursor()
    c.execute('''
        UPDATE complaints 
        SET status = 'İnceleniyor' 
        WHERE status = 'İşleme Alındı' OR status = 'İnceleme Yapılıyor'
    ''')
    conn.commit()
    conn.close()

# Ana kodun başında çağırın
update_existing_statuses()

def update_existing_model_names():
    conn = sqlite3.connect('complaints.db')
    c = conn.cursor()
    try:
        c.execute('''
            UPDATE complaints 
            SET model_used = 'MultinomialNB' 
            WHERE model_used = 'mb' OR model_used IS NULL OR model_used = '';
            
            UPDATE complaints 
            SET model_used = 'SGD Classifier' 
            WHERE model_used = 'sgd';
            
            UPDATE complaints 
            SET model_used = 'Logistic Regression' 
            WHERE model_used = 'lr';
        ''')
        conn.commit()
    except Exception as e:
        print(f"Model güncelleme hatası: {e}")
        conn.rollback()
    finally:
        conn.close()

# Ana kodun başında çağır
update_existing_model_names()

# Mevcut kayıtları güncellemek için yeni fonksiyon
def update_model_names():
    conn = sqlite3.connect('complaints.db')
    c = conn.cursor()
    try:
        # Önce mevcut model isimlerini kontrol et
        c.execute('SELECT DISTINCT model_used FROM complaints')
        current_models = c.fetchall()
        print("Mevcut model isimleri:", current_models)  # Debug için
        
        # Model isimlerini güncelle
        c.execute('''
            UPDATE complaints 
            SET model_used = 'MultinomialNB' 
            WHERE model_used = 'mb' OR model_used = '';
        ''')
        c.execute('''
            UPDATE complaints 
            SET model_used = 'SGD Classifier' 
            WHERE model_used = 'sgd';
        ''')
        c.execute('''
            UPDATE complaints 
            SET model_used = 'Logistic Regression' 
            WHERE model_used = 'lr';
        ''')
        c.execute('''
            UPDATE complaints 
            SET model_used = 'Deep Learning' 
            WHERE model_used = 'dl';
        ''')
        conn.commit()
        
        # Güncelleme sonrası kontrol
        c.execute('SELECT DISTINCT model_used FROM complaints')
        updated_models = c.fetchall()
        print("Güncellenmiş model isimleri:", updated_models)  # Debug için
        
    except Exception as e:
        print(f"Model güncelleme hatası: {e}")
        conn.rollback()
    finally:
        conn.close()

# Debug fonksiyonu
def debug_complaints():
    conn = sqlite3.connect('complaints.db')
    c = conn.cursor()
    try:
        c.execute('''
            SELECT complaint_number, model_used, category, status
            FROM complaints
            ORDER BY created_at DESC
            LIMIT 10
        ''')
        complaints = c.fetchall()
        print("\nSon 10 şikayet:")
        for complaint in complaints:
            print(f"Şikayet #{complaint[0]}: Model={complaint[1]}, Kategori={complaint[2]}, Durum={complaint[3]}")
            
        c.execute('SELECT COUNT(*), model_used FROM complaints GROUP BY model_used')
        counts = c.fetchall()
        print("\nModel bazlı şikayet sayıları:")
        for count in counts:
            print(f"{count[1]}: {count[0]} şikayet")
            
    except Exception as e:
        print(f"Debug hatası: {e}")
    finally:
        conn.close()

# Ana kodun sonuna ekle
if __name__ == "__main__":
    update_model_names()  # Mevcut kayıtları güncelle
    debug_complaints()    # Debug bilgilerini göster
