import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import time
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import hashlib

# ---------- Page Config ----------
st.set_page_config(
    page_title="Traffic AI Dashboard",
    page_icon="🚦",
    layout="wide"
)

# ---------- UI Styling ----------
st.markdown("""
<style>
.stButton>button {
    border-radius: 10px;
    height: 42px;
    font-size: 16px;
}
[data-testid="stMetric"] {
    background-color: #262730;
    padding: 15px;
    border-radius: 10px;
}
.center-box {
    max-width: 400px;
    margin: auto;
}
</style>
""", unsafe_allow_html=True)

# ---------- SESSION ----------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ---------- DATABASE ----------
def connect_db():
    return sqlite3.connect("users.db")

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username, password):
    conn = connect_db()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users VALUES (?, ?)", (username, hash_password(password)))
        conn.commit()
        return True
    except:
        return False

def login_user(username, password):
    conn = connect_db()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? AND password=?",
              (username, hash_password(password)))
    return c.fetchone()

# ---------- AUTH SCREEN ----------
def auth_screen():
    st.markdown('<div class="center-box">', unsafe_allow_html=True)

    st.title("🔐 Traffic Dashboard")
    st.markdown("### Login or Register")

    tab1, tab2 = st.tabs(["Login", "Register"])

    # LOGIN
    with tab1:
        user = st.text_input("Username", key="login_user")
        pwd = st.text_input("Password", type="password", key="login_pwd")

        if st.button("Login", key="login_btn"):
            if login_user(user, pwd):
                st.session_state.logged_in = True
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid credentials")

    # REGISTER
    with tab2:
        new_user = st.text_input("New Username", key="reg_user")
        new_pwd = st.text_input("New Password", type="password", key="reg_pwd")

        if st.button("Register", key="reg_btn"):
            if create_user(new_user, new_pwd):
                st.success("Account created! Now login.")
            else:
                st.error("User already exists")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- MAIN APP ----------
def main_app():
    model = joblib.load("saved_models/Random_Forest.pkl")

    # Header
    col1, col2 = st.columns([8,1])

    with col1:
        st.title("🚦 Traffic Prediction Dashboard")

    with col2:
        if st.button("🚪 Logout"):
            st.warning("Logged out successfully")
            st.session_state.logged_in = False
            st.rerun()

    st.markdown("### Smart traffic Volume prediction using Machine Learning 🚀")

    tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Analytics", "📁 Batch"])

    # ===================== TAB 1 =====================
    with tab1:
        st.subheader("🔢 Input Features")

        col1, col2 = st.columns(2)

        with col1:
            hour = st.slider("⏰ Hour", 0, 23, 12)
            day = st.slider("📅 Day", 1, 31, 15)

        with col2:
            month = st.slider("🗓️ Month", 1, 12, 6)
            weekday = st.slider("📆 Weekday (0=Mon, 6=Sun)", 0, 6, 2)

        is_rush = st.toggle("🚦 Rush Hour")

        st.markdown("---")

        if st.button("🚀 Predict Now"):

            with st.spinner("Analyzing traffic patterns... ⏳"):
                time.sleep(1)

                features = np.array([[hour, day, month, weekday, int(is_rush)]])
                prediction = model.predict(features)[0]

            st.subheader("📊 Prediction Result")

            # 🔥 smoother animation
            placeholder = st.empty()
            for i in np.linspace(1, prediction, 100):
                placeholder.metric("🚗 Traffic Volume", round(i, 2))
                time.sleep(0.01)

            placeholder.metric("🚗 Traffic Volume", round(prediction, 2))

            if prediction < 2000:
                st.success("🟢 Smooth Traffic Flow")
            elif prediction < 5000:
                st.warning("🟡 Moderate Traffic")
            else:
                st.error("🔴 Heavy Traffic")

    # ===================== TAB 2 =====================
    with tab2:
        st.subheader("📊 Traffic Analytics")

        if hasattr(model, "feature_importances_"):
            df_imp = pd.DataFrame({
                "Feature": ["Hour", "Day", "Month", "Weekday", "Rush"],
                "Importance": model.feature_importances_
            })

            fig = px.bar(df_imp, x="Feature", y="Importance", color="Importance")
            st.plotly_chart(fig, use_container_width=True)

        hours = list(range(24))
        sample = np.array([[h, 15, 6, 2, 1] for h in hours])
        preds = model.predict(sample)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hours, y=preds, mode='lines+markers'))

        st.plotly_chart(fig, use_container_width=True)

    # ===================== TAB 3 =====================
    with tab3:
        st.subheader("📁 Batch Prediction")

        file = st.file_uploader("Upload CSV", type=["csv"])

        if file:
            data = pd.read_csv(file)

            if "date_time" in data.columns:
                data['date_time'] = pd.to_datetime(data['date_time'], dayfirst=True)
                data['hour'] = data['date_time'].dt.hour
                data['day'] = data['date_time'].dt.day
                data['month'] = data['date_time'].dt.month
                data['weekday'] = data['date_time'].dt.weekday
                data['is_rush'] = data['hour'].apply(
                    lambda x: 1 if x in [7,8,9,17,18,19] else 0
                )

            if {"hour", "day", "month", "weekday", "is_rush"}.issubset(data.columns):
                preds = model.predict(
                    data[["hour", "day", "month", "weekday", "is_rush"]]
                )
                data["Prediction"] = preds

                st.dataframe(data)

                st.download_button(
                    "⬇ Download Results",
                    data.to_csv(index=False),
                    "predictions.csv"
                )
            else:
                st.error("CSV must contain required columns")

# ---------- ROUTING ----------
if not st.session_state.logged_in:
    auth_screen()
else:
    main_app()
