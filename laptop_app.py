import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import base64


st.set_page_config(
    page_title="Laptop Price Prediction",
    page_icon="💻",
    layout="centered"
)

st.markdown("""
<style>

/* Sidebar Background */
section[data-testid="stSidebar"] {
    background: linear-gradient(-45deg,#ff6ec4,#7873f5,#42e695,#f9f871);
    background-size: 400% 400%;
    padding-top: 20px;
    animation: gradientBG 15s ease infinite;
}

@keyframes gradientBG{
            0%{background-position: 0% 50%;}
            50%{background-position: 100% 50%;}
            100%{background-position: 0% 50%;}
        }

/* Sidebar Title */
.sidebar-title {
    font-size: 22px;
    font-weight: 700;
    color: white;
    margin-bottom: 15px;
}

/* Laptop Card */
.laptop-card {
    background: #1e293b;
    padding: 10px;
    border-radius: 15px;
    margin-bottom: 18px;
    transition: 0.3s ease-in-out;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
}

.laptop-card:hover {
    transform: scale(1.04);
    box-shadow: 0 0 18px #ff8c00;
}

/* Brand Name */
.brand-text {
    color: #ffffff;
    font-weight: 600;
    font-size: 16px;
    margin-bottom: 8px;
}

/* Image Styling */
.laptop-img {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-title">🔥 Laptop Deals</div>', unsafe_allow_html=True)

def sidebar_card(brand, link, image):
    st.sidebar.markdown(f"""
    <a href="{link}" target="_blank" style="text-decoration:none;">
        <div class="laptop-card">
            <div class="brand-text">{brand}</div>
            <img class="laptop-img" src="{image}" width="100%">
        </div>
    </a>
    """, unsafe_allow_html=True)


# Add Cards
sidebar_card("Lenovo", "https://www.amazon.in/s?k=Lenovo+Laptops",
             "https://5.imimg.com/data5/DG/HJ/MY-43212347/lenovo-laptop-500x500.jpg")

sidebar_card("Samsung", "https://www.amazon.in/s?k=Samsung+Laptops",
             "https://rukminim2.flixcart.com/image/480/640/xif0q/computer/j/3/f/-original-imahfu5nynahgvfy.jpeg?q=90")

sidebar_card("LG", "https://www.amazon.in/s?k=LG+Laptops",
             "https://m.media-amazon.com/images/I/71a7M-Sd1XL._AC_UF1000,1000_QL80_.jpg")

sidebar_card("Dell", "https://www.amazon.in/s?k=Dell+Laptops",
             "https://media-ik.croma.com/prod/https://media.tatacroma.com/Croma%20Assets/Computers%20Peripherals/Laptop/Images/271916_0_ezz3ms.png")

sidebar_card("Asus", "https://www.amazon.in/s?k=Asus+Laptops",
             "https://digitalstore.com.co/wp-content/uploads/2024/05/Portatil-Asus-ExpertBook-B1402CB-NK3018X-%E2%80%93-Intel-Core-i5-1235U-%E2%80%93-8GB-RAM-%E2%80%93-SSD-512GB-%E2%80%93-15P-W10-Pro.jpg")

sidebar_card("HP", "https://www.amazon.in/s?k=HP+Laptops",
             "https://rukminim2.flixcart.com/image/480/640/xif0q/computer/w/e/j/notebook-thin-and-light-laptop-hp-original-imah2uwjhzdhxg3u.jpeg?q=90")

sidebar_card("MSI", "https://www.amazon.in/s?k=MSI+Laptops",
             "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQVAuv7RMkXBPTjVKj81SMIHhe0bbZQR51w-Q&s")

sidebar_card("Apple", "https://www.amazon.in/s?k=Apple+MacBook",
             "https://images.unsplash.com/photo-1517336714731-489689fd1ca8")

sidebar_card("Acer", "https://www.amazon.in/s?k=Acer+Laptops",
             "https://m.media-amazon.com/images/I/513p8BwV-RL._AC_UF1000,1000_QL80_.jpg")

st.sidebar.caption("🛒 Click any brand to explore latest offers on Amazon")

def get_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img_base64 = get_base64("laptop.png")

st.markdown(f"""
<style>

.stApp {{
    # background: radial-gradient(circle at top left, #0f2027, #203a43, #2c5364);
    color: white;
}}

/* Transparent Laptop Image */
.stApp::before {{
    content: "";
    position: fixed;
    top: 30%;
    right: 15%;
    width: 650px;
    height: 650px;
    background-image: url("data:image/png;base64,{img_base64}");
    background-repeat: no-repeat;
    background-size: contain;
    opacity: 0.7;
    
    z-index: 1;
}}

h1 {{
    text-align: center;
}}

.block-container {{
    padding-top: 2rem;
    padding-bottom: 2rem;
    position: relative;
    z-index: 1;
}}

.stButton>button {{
    background: linear-gradient(90deg, #ff8c00, #ff2e63);
    color: white;
    font-size: 18px;
    border-radius: 30px;
    padding: 12px 25px;
    border: none;
    transition: 0.3s;
}}

.stButton>button:hover {{
    box-shadow: 0 0 25px #ff8c00;
    transform: scale(1.05);
}}

</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    pipeline = joblib.load("pipeline.pkl")
    return model, pipeline

model, pipeline = load_model()
st.markdown("""<style>
            div[data-testid="stImageContainer"]{
                display: flex;
                justify-content: center;
                margin-bottom: 10px;
                padding: 10px 20px;
                padding-top: 10px;
                height: 40vh;
                width: 10px 20px;
            }
            </style>""",unsafe_allow_html=True)
st.image("laptop_logo.png",width=500)
# st.markdown("<h1>💻 Laptop Price Prediction</h1>", unsafe_allow_html=True) 
st.markdown(
    "<p style='text-align:center; font-size:18px;'>"
    "This application predicts the <b>Price of Laptop</b> using a trained Machine Learning Model."
    "</p>",
    unsafe_allow_html=True
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Brand",
        ["Lenovo","Samsung","LG","Dell","Asus","HP","MSI","Apple","Acer"], index=2)

    cpu_brand = st.selectbox("CPU Brand",
        ["AMD","Apple Silicon","Intel"], index=0)

    ram_gb = st.selectbox("RAM (GB)",
        [4,8,16,24,32,64], index=3)

    storage_gb = st.selectbox("Storage (GB)",
        [256,512,1024,2048], index=1)

    storage_type = st.selectbox("Storage Type",
        ["SSD","HDD","Hybrid"], index=0)

    gpu_brand = st.selectbox("GPU Brand",
        ["AMD","Intel","NVIDIA","M-Series"], index=1)

    screen_size_inch = st.number_input(
        "Screen Size (inch)", 11.0, 18.0, 17.5, step=0.1)

with col2:
    battery_wh = st.selectbox("Battery (Wh)",
        [35,45,55,65,80,99], index=3)

    release_year = st.selectbox("Release Year",
        [2016,2017,2018,2019,2020,2021,2022,2023], index=5)

    os = st.selectbox("Operating System",
        ["Ubuntu","macOS","ChromeOS","Windows 10","Windows 11","Fedora"], index=0)

    region = st.selectbox("Region",
        ["India","SE Asia","Europe","Middle East","USA"], index=2)

    dedicated_gpu_vram_gb = st.selectbox("GPU VRAM (GB)",
        [0,2,4,6,8,12], index=2)

    screen_type = st.selectbox("Screen Type",
        ["IPS","VA","TN","OLED"], index=0)

    num_reviews = st.number_input("Number of Reviews", 0, 10000, 324)
    
resolution = st.selectbox("Resolution",
        ["1366x768","1920x1080","2560x1440","3840x2160"], index=2)

col3,col4 = st.columns(2)

with col3:
    weight_kg = st.slider("Weight (kg)", 1.0, 4.0, 2.9, step=0.01)
    
    touchscreen = 1 if st.toggle("Touchscreen", value=True) else 0
    
    backlit_keyboard = 1 if st.toggle("Backlit Keyboard", value=True) else 0
    
with col4:
    user_rating = st.slider("User Rating", 1.0, 5.0, 3.7, step=0.1)
    
    fingerprint_reader = 1 if st.toggle("Fingerprint Reader") else 0
    
    wifi6_supported = 1 if st.toggle("WiFi 6 Support", value=True) else 0

st.divider()

predict_btn = st.button("🚀 Predict Laptop Price")

if predict_btn:
    try:
        input_data = pd.DataFrame([{
            'brand': brand,
            'cpu_brand': cpu_brand,
            'ram_gb': ram_gb,
            'storage_gb': storage_gb,
            'storage_type': storage_type,
            'gpu_brand': gpu_brand,
            'dedicated_gpu_vram_gb': dedicated_gpu_vram_gb,
            'screen_size_inch': screen_size_inch,
            'screen_type': screen_type,
            'resolution': resolution,
            'weight_kg': weight_kg,
            'battery_wh': battery_wh,
            'release_year': release_year,
            'os': os,
            'region': region,
            'user_rating': user_rating,
            'num_reviews': num_reviews,
            'touchscreen': touchscreen,
            'backlit_keyboard': backlit_keyboard,
            'fingerprint_reader': fingerprint_reader,
            'wifi6_supported': wifi6_supported
        }])


        transformed_data = pipeline.transform(input_data)
        prediction = model.predict(transformed_data)[0]

        with st.spinner("Analyzing Specifications..."):
            time.sleep(2)

        st.markdown(f"""
        <div style="
        background: linear-gradient(90deg,#ff8c00,#ff2e63);
        padding:20px;
        border-radius:15px;
        text-align:center;
        font-size:22px;
        font-weight:bold;
        color:white;">
        💰 Predicted Laptop Price: ₹ {round(prediction)}
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error occurred: {e}")

st.divider()

st.caption(
    "🧠 ML-powered Laptop Price Prediction built with "
    "Python, NumPy, Pandas, Scikit-learn, Joblib, and Streamlit."
)
