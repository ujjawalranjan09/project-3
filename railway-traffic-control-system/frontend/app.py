import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

st.set_page_config(
    page_title="Railway Traffic Control System",
    page_icon="🚂",
    layout="wide"
)

# --- Configuration ---
st.sidebar.title("Settings")
api_base_url = st.sidebar.text_input("API Base URL", value="http://localhost:5000/api")
api_key = st.sidebar.text_input("API Key", value="test-api-key", type="password")

headers = {"X-API-Key": api_key}

# --- Helper Functions ---
def get_kpis():
    try:
        current_headers = {"X-API-Key": api_key}
        response = requests.get(f"{api_base_url}/metrics/kpi", headers=current_headers)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Error fetching KPIs: {e}")
    return None

def predict_conflict(data):
    try:
        response = requests.post(f"{api_base_url}/predict/conflict", json=data, headers=headers, timeout=5)
        return response.json(), response.status_code
    except Exception as e:
        return {"error": str(e)}, 500

def predict_delay(data):
    try:
        response = requests.post(f"{api_base_url}/predict/delay", json=data, headers=headers, timeout=5)
        return response.json(), response.status_code
    except Exception as e:
        return {"error": str(e)}, 500

def simulate_scenario(baseline, modifications):
    try:
        data = {"baseline": baseline, "modifications": modifications}
        response = requests.post(f"{api_base_url}/simulate", json=data, headers=headers, timeout=5)
        return response.json(), response.status_code
    except Exception as e:
        return {"error": str(e)}, 500

# --- Header ---
st.title("🚂 Railway Traffic Control System")
st.markdown("### AI-Powered Decision Support Dashboard")

# --- KPI Section ---
kpis = get_kpis()
if kpis:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Throughput", f"{kpis['throughput']['trains_per_hour']} t/h",
                  f"{kpis['throughput']['percentage']}% capacity")

    with col2:
        st.metric("Punctuality", f"{kpis['punctuality']['on_time_percentage']}%",
                  f"Target: {kpis['punctuality']['target']}%")

    with col3:
        st.metric("Avg Delay", f"{kpis['average_delay']['current']} min",
                  f"Target: {kpis['average_delay']['target']} min", delta_color="inverse")

    with col4:
        st.metric("Active Conflicts", kpis['conflicts']['pending'],
                  f"Resolved Today: {kpis['conflicts']['resolved']}")

st.divider()

# --- Analysis Section ---
col_left, col_right = st.columns(2)

with col_left:
    st.header("Conflict Detection")
    with st.form("conflict_form"):
        c1, c2 = st.columns(2)
        with c1:
            trains = st.number_input("Trains in Section", value=30, min_value=0, max_value=200)
            platforms = st.number_input("Available Platforms", value=3, min_value=0, max_value=50)
            util = st.slider("Platform Utilization (%)", 0.0, 100.0, 90.0)
            peak = st.selectbox("Peak Hour", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        with c2:
            weather = st.slider("Weather Severity", 0.0, 1.0, 0.2)
            rain = st.number_input("Rainfall (mm)", value=0.0, min_value=0.0, max_value=500.0)
            fog = st.slider("Fog Intensity", 0.0, 1.0, 0.0)
            temp = st.number_input("Temperature (°C)", value=25.0, min_value=-50.0, max_value=60.0)

        submitted = st.form_submit_button("Analyze Conflict Risk", use_container_width=True)

        if submitted:
            data = {
                "trains_in_section": trains,
                "available_platforms": platforms,
                "platform_utilization": util,
                "weather_severity": weather,
                "rainfall_mm": rain,
                "fog_intensity": fog,
                "temperature_c": temp,
                "is_peak_hour": peak
            }
            st.session_state.current_data = data
            result, status = predict_conflict(data)

            if status == 200:
                st.session_state.conflict_result = result
            else:
                st.error(f"Error: {result.get('error', 'Unknown error')}")

    if "conflict_result" in st.session_state:
        res = st.session_state.conflict_result
        risk = res['risk_level']
        color = "red" if risk == "HIGH" else "orange" if risk == "MEDIUM" else "green"
        st.markdown(f"### Risk Level: :{color}[{risk}]")
        st.write(f"**Probability:** {res['conflict_probability'] * 100:.1f}%")

        if res.get('recommendations'):
            st.write("**Recommendations:**")
            for rec in res['recommendations']:
                st.info(f"**{rec['priority']}:** {rec['action']} - {rec['details']}")

with col_right:
    st.header("Delay Prediction")
    if "current_data" not in st.session_state:
        st.warning("Please run Conflict Analysis first to load data.")
    else:
        if st.button("Predict Delay (Same Inputs)", use_container_width=True):
            result, status = predict_delay(st.session_state.current_data)
            if status == 200:
                st.session_state.delay_result = result
            else:
                st.error(f"Error: {result.get('error', 'Unknown error')}")

    if "delay_result" in st.session_state:
        res = st.session_state.delay_result
        severity = res['severity']
        color = "red" if severity == "CRITICAL" else "orange" if severity == "SIGNIFICANT" else "green"
        st.markdown(f"### Severity: :{color}[{severity}]")
        st.write(f"**Predicted Delay:** {res['predicted_delay_minutes']} minutes")
        st.write(f"**Impact:** {res['impact_description']}")

        if res.get('mitigation_strategies'):
            st.write("**Mitigation Strategies:**")
            for mit in res['mitigation_strategies']:
                with st.expander(mit['strategy']):
                    st.write(f"**Implementation:** {mit['implementation']}")
                    st.write(f"**Expected Reduction:** {mit['expected_reduction']}")

st.divider()

# --- Simulation Section ---
st.header("What-If Scenario Simulation")
if "current_data" not in st.session_state:
    st.warning("Please run Conflict Analysis first to set baseline.")
else:
    s1, s2, s3 = st.columns([2, 2, 1])
    with s1:
        sim_trains = st.number_input("Modify Trains", value=st.session_state.current_data['trains_in_section'])
    with s2:
        sim_platforms = st.number_input("Modify Platforms", value=st.session_state.current_data['available_platforms'])
    with s3:
        st.write("##")
        run_sim = st.button("Run Simulation", use_container_width=True)

    if run_sim:
        mods = {}
        if sim_trains != st.session_state.current_data['trains_in_section']:
            mods['trains_in_section'] = sim_trains
        if sim_platforms != st.session_state.current_data['available_platforms']:
            mods['available_platforms'] = sim_platforms

        if not mods:
            st.info("No modifications made to baseline.")
        else:
            result, status = simulate_scenario(st.session_state.current_data, mods)
            if status == 200:
                st.success(f"Simulation Complete!")
                c1, c2, c3 = st.columns(3)
                c1.metric("Baseline Delay", f"{result['baseline']['predicted_delay_minutes']} min")
                c2.metric("Modified Delay", f"{result['modified_scenario']['predicted_delay_minutes']} min")

                reduction = result['delay_reduction_minutes']
                c3.metric("Improvement", f"{reduction} min", f"{result['improvement_percentage']}%")
            else:
                st.error(f"Error: {result.get('error', 'Unknown error')}")

st.divider()

# --- Monitoring Charts ---
st.header("Real-time Monitoring")
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    # Throughput Trend
    df_throughput = pd.DataFrame({
        'Time': ['6:00', '9:00', '12:00', '15:00', '18:00', '21:00'],
        'Trains/Hour': [35, 48, 42, 45, 52, 38]
    })
    fig1 = px.line(df_throughput, x='Time', y='Trains/Hour', title="Throughput Trend", markers=True)
    st.plotly_chart(fig1, use_container_width=True)

with chart_col2:
    # Weekly Delay Trend
    df_delay = pd.DataFrame({
        'Day': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
        'Avg Delay (min)': [12, 8, 10, 7, 9, 6, 5]
    })
    fig2 = px.bar(df_delay, x='Day', y='Avg Delay (min)', title="Weekly Delay Trend", color_discrete_sequence=['#e74c3c'])
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.caption(f"Railway Traffic Control System v1.1 | Last synced: {datetime.now().strftime('%H:%M:%S')}")
