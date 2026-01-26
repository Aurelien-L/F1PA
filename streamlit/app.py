"""
F1PA - Formula 1 Predictive Assistant
Streamlit Dashboard

Interface for lap time predictions using the F1PA ML model.
"""
import streamlit as st
import requests
from requests.auth import HTTPBasicAuth
import pandas as pd

from config import (
    API_BASE_URL, API_EXTERNAL_URL, API_USERNAME, API_PASSWORD,
    MLFLOW_URL, GITHUB_URL, EVIDENTLY_URL,
    DEFAULT_TEMP, DEFAULT_RHUM, DEFAULT_PRES,
    DEFAULT_LAP_NUMBER, DEFAULT_YEAR,
    SPEED_MIN, SPEED_MAX,
    DEFAULT_ST_SPEED, DEFAULT_I1_SPEED, DEFAULT_I2_SPEED
)

# =============================================================================
# PAGE CONFIG & STYLING
# =============================================================================

st.set_page_config(
    page_title="F1PA - Lap Time Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# F1-inspired CSS (Black, White, Red)
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #15151E;
    }

    /* Headers */
    h1, h2, h3 {
        color: #FFFFFF !important;
    }

    /* Red accent for important elements */
    .stButton > button {
        background-color: #E10600 !important;
        color: white !important;
        border: none !important;
        border-radius: 4px !important;
        font-weight: bold !important;
        padding: 0.5rem 2rem !important;
    }

    .stButton > button:hover {
        background-color: #FF1E00 !important;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1E1E2E;
        padding: 0.5rem;
        border-radius: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #2D2D3D;
        color: #FFFFFF;
        border-radius: 4px;
        padding: 0.5rem 1rem;
    }

    .stTabs [aria-selected="true"] {
        background-color: #E10600 !important;
    }

    /* Cards/Containers */
    .prediction-card {
        background: linear-gradient(135deg, #1E1E2E 0%, #2D2D3D 100%);
        border-radius: 12px;
        padding: 2rem;
        border-left: 4px solid #E10600;
        margin: 1rem 0;
    }

    .metric-card {
        background-color: #1E1E2E;
        border-radius: 8px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid #3D3D4D;
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #E10600;
    }

    .metric-label {
        color: #AAAAAA;
        font-size: 0.9rem;
        text-transform: uppercase;
    }

    /* Driver photo container */
    .driver-photo {
        border-radius: 50%;
        border: 3px solid #E10600;
        width: 150px;
        height: 150px;
        object-fit: cover;
    }

    /* Result display */
    .lap-time-result {
        font-size: 4rem;
        font-weight: bold;
        color: #FFFFFF;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #E10600, #FF4444);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* Selectbox styling */
    .stSelectbox label, .stSlider label, .stNumberInput label {
        color: #FFFFFF !important;
    }

    /* Link cards */
    .link-card {
        background-color: #1E1E2E;
        border-radius: 8px;
        padding: 1.5rem;
        border: 1px solid #3D3D4D;
        transition: all 0.3s ease;
    }

    .link-card:hover {
        border-color: #E10600;
        transform: translateY(-2px);
    }

    /* Info boxes */
    .info-box {
        background-color: #2D2D3D;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }

    /* Comparison table */
    .comparison-row {
        display: flex;
        justify-content: space-between;
        padding: 0.5rem 0;
        border-bottom: 1px solid #3D3D4D;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# API HELPERS
# =============================================================================

def api_request(endpoint: str, method: str = "GET", json_data: dict = None) -> dict:
    """Make authenticated API request."""
    url = f"{API_BASE_URL}{endpoint}"
    auth = HTTPBasicAuth(API_USERNAME, API_PASSWORD)

    try:
        if method == "GET":
            response = requests.get(url, auth=auth, timeout=10)
        elif method == "POST":
            response = requests.post(url, auth=auth, json=json_data, timeout=10)
        else:
            raise ValueError(f"Unsupported method: {method}")

        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to API at {API_BASE_URL}. Is the API running?")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"API Error: {e}")
        return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None


@st.cache_data(ttl=300)
def get_drivers() -> list:
    """Fetch drivers from API."""
    data = api_request("/data/drivers")
    return data if data else []


@st.cache_data(ttl=300)
def get_circuits() -> list:
    """Fetch circuits from API."""
    data = api_request("/data/circuits")
    return data if data else []


@st.cache_data(ttl=60)
def get_model_info() -> dict:
    """Fetch model info from API."""
    return api_request("/predict/model")


@st.cache_data(ttl=300)
def get_circuit_avg_laptime(circuit_key: int) -> float:
    """Fetch circuit average lap time."""
    data = api_request(f"/data/circuits/{circuit_key}/avg-laptime")
    if data:
        return data.get("avg_laptime_seconds", 90.0)
    return 90.0


@st.cache_data(ttl=300)
def get_driver_stats(driver_number: int) -> dict:
    """Fetch driver statistics (laps) to compute averages."""
    data = api_request(f"/data/drivers/{driver_number}/laps?limit=100")
    if data and len(data) > 0:
        df = pd.DataFrame(data)
        return {
            "avg_laptime": df["lap_duration"].mean(),
            "avg_st_speed": df["st_speed"].mean() if "st_speed" in df else DEFAULT_ST_SPEED,
            "avg_i1_speed": df["i1_speed"].mean() if "i1_speed" in df else DEFAULT_I1_SPEED,
            "avg_i2_speed": df["i2_speed"].mean() if "i2_speed" in df else DEFAULT_I2_SPEED,
        }
    return {
        "avg_laptime": 90.0,
        "avg_st_speed": DEFAULT_ST_SPEED,
        "avg_i1_speed": DEFAULT_I1_SPEED,
        "avg_i2_speed": DEFAULT_I2_SPEED,
    }


def make_prediction(features: dict) -> dict:
    """Make lap time prediction."""
    return api_request("/predict/lap", method="POST", json_data={"features": features})


# =============================================================================
# HEADER
# =============================================================================

def render_header():
    """Render the app header."""
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h1 style="margin: 0; font-size: 3rem;">🏎️ F1PA</h1>
            <p style="color: #AAAAAA; margin: 0;">Formula 1 Predictive Assistant</p>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# TAB 1: PREDICTION
# =============================================================================

def render_prediction_tab():
    """Render the prediction interface."""

    # Fetch data
    drivers = get_drivers()
    circuits = get_circuits()

    if not drivers or not circuits:
        st.warning("Cannot load data from API. Please ensure the API is running.")
        return

    # Create selection options
    driver_options = {f"{d['full_name']} ({d['name_acronym']})": d for d in drivers}
    circuit_options = {f"{c['circuit_short_name']} - {c['country_name']}": c for c in circuits}

    # Layout: Two columns
    col_left, col_right = st.columns([1, 1])

    # ===================
    # LEFT: Selection
    # ===================
    with col_left:
        st.markdown("### Select Driver & Circuit")

        # Driver selection
        selected_driver_name = st.selectbox(
            "Driver",
            options=list(driver_options.keys()),
            index=0,
            key="driver_select"
        )
        selected_driver = driver_options[selected_driver_name]

        # Display driver photo and info
        col_photo, col_info = st.columns([1, 2])
        with col_photo:
            if selected_driver.get("headshot_url"):
                st.image(
                    selected_driver["headshot_url"],
                    width=120,
                    caption=selected_driver["name_acronym"]
                )
            else:
                st.markdown(f"""
                <div style="width: 120px; height: 120px; background: #2D2D3D;
                            border-radius: 50%; display: flex; align-items: center;
                            justify-content: center; font-size: 2rem; color: #E10600;">
                    {selected_driver["name_acronym"]}
                </div>
                """, unsafe_allow_html=True)

        with col_info:
            team_color = selected_driver.get("team_colour", "E10600")
            st.markdown(f"""
            <div class="info-box">
                <div style="color: #{team_color}; font-weight: bold; font-size: 1.2rem;">
                    #{selected_driver['driver_number']} {selected_driver['full_name']}
                </div>
                <div style="color: #AAAAAA;">{selected_driver.get('team_name', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Circuit selection
        selected_circuit_name = st.selectbox(
            "Circuit",
            options=list(circuit_options.keys()),
            index=0,
            key="circuit_select"
        )
        selected_circuit = circuit_options[selected_circuit_name]

        st.markdown(f"""
        <div class="info-box">
            <div style="color: #FFFFFF; font-weight: bold;">
                {selected_circuit['circuit_short_name']}
            </div>
            <div style="color: #AAAAAA;">
                {selected_circuit.get('location', '')} - {selected_circuit['country_name']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ===================
    # RIGHT: Parameters
    # ===================
    with col_right:
        st.markdown("### Prediction Parameters")

        # Get driver stats for defaults
        driver_stats = get_driver_stats(selected_driver["driver_number"])
        circuit_avg = get_circuit_avg_laptime(selected_circuit["circuit_key"])

        # Lap number
        lap_number = st.slider(
            "Lap Number",
            min_value=1,
            max_value=78,
            value=DEFAULT_LAP_NUMBER,
            help="Which lap of the race to predict"
        )

        # Expandable: Advanced parameters
        with st.expander("Advanced Parameters", expanded=False):
            st.markdown("**Speed Settings (km/h)**")
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                st_speed = st.number_input(
                    "Speed Trap",
                    min_value=SPEED_MIN,
                    max_value=SPEED_MAX,
                    value=float(driver_stats["avg_st_speed"]),
                    step=1.0
                )
            with col_s2:
                i1_speed = st.number_input(
                    "Intermediate 1",
                    min_value=SPEED_MIN,
                    max_value=SPEED_MAX,
                    value=float(driver_stats["avg_i1_speed"]),
                    step=1.0
                )
            with col_s3:
                i2_speed = st.number_input(
                    "Intermediate 2",
                    min_value=SPEED_MIN,
                    max_value=SPEED_MAX,
                    value=float(driver_stats["avg_i2_speed"]),
                    step=1.0
                )

            st.markdown("**Weather Conditions**")
            col_w1, col_w2, col_w3 = st.columns(3)
            with col_w1:
                temp = st.number_input("Temperature (°C)", value=DEFAULT_TEMP, step=1.0)
            with col_w2:
                rhum = st.number_input("Humidity (%)", value=DEFAULT_RHUM, step=5.0)
            with col_w3:
                pres = st.number_input("Pressure (hPa)", value=DEFAULT_PRES, step=1.0)

        # Use defaults if not expanded
        if 'st_speed' not in dir():
            st_speed = driver_stats["avg_st_speed"]
            i1_speed = driver_stats["avg_i1_speed"]
            i2_speed = driver_stats["avg_i2_speed"]
            temp = DEFAULT_TEMP
            rhum = DEFAULT_RHUM
            pres = DEFAULT_PRES

        # Calculate driver_perf_score
        driver_avg_laptime = driver_stats["avg_laptime"]
        driver_perf_score = driver_avg_laptime - circuit_avg

        st.markdown("<br>", unsafe_allow_html=True)

        # Predict button
        if st.button("🏁 Predict Lap Time", use_container_width=True):
            # Build features
            features = {
                "driver_number": selected_driver["driver_number"],
                "circuit_key": selected_circuit["circuit_key"],
                "st_speed": st_speed,
                "i1_speed": i1_speed,
                "i2_speed": i2_speed,
                "temp": temp,
                "rhum": rhum,
                "pres": pres,
                "lap_number": lap_number,
                "year": DEFAULT_YEAR,
                "circuit_avg_laptime": circuit_avg,
                "driver_avg_laptime": driver_avg_laptime,
                "driver_perf_score": driver_perf_score
            }

            with st.spinner("Predicting..."):
                result = make_prediction(features)

            if result:
                st.session_state["prediction_result"] = result
                st.session_state["prediction_context"] = {
                    "driver": selected_driver,
                    "circuit": selected_circuit,
                    "circuit_avg": circuit_avg,
                    "driver_avg": driver_avg_laptime,
                    "lap_number": lap_number
                }

    # ===================
    # RESULT DISPLAY
    # ===================
    if "prediction_result" in st.session_state:
        result = st.session_state["prediction_result"]
        context = st.session_state["prediction_context"]

        st.markdown("---")
        st.markdown("### Prediction Result")

        col_main, col_compare = st.columns([2, 1])

        with col_main:
            st.markdown(f"""
            <div class="prediction-card">
                <div style="text-align: center;">
                    <div style="color: #AAAAAA; text-transform: uppercase; letter-spacing: 2px;">
                        Predicted Lap Time
                    </div>
                    <div class="lap-time-result">
                        {result['lap_duration_formatted']}
                    </div>
                    <div style="color: #AAAAAA; font-size: 0.9rem;">
                        {result['lap_duration_seconds']:.3f} seconds
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_compare:
            st.markdown("**Comparison**")

            predicted = result['lap_duration_seconds']
            circuit_avg = context['circuit_avg']
            driver_avg = context['driver_avg']

            # Format times
            def format_time(seconds):
                mins = int(seconds // 60)
                secs = seconds % 60
                return f"{mins}:{secs:06.3f}"

            # Delta calculation
            delta_circuit = predicted - circuit_avg
            delta_driver = predicted - driver_avg

            st.markdown(f"""
            <div class="info-box">
                <div style="color: #AAAAAA; font-size: 0.8rem;">Circuit Average</div>
                <div style="color: #FFFFFF; font-size: 1.1rem;">{format_time(circuit_avg)}</div>
                <div style="color: {'#00FF00' if delta_circuit < 0 else '#FF4444'}; font-size: 0.9rem;">
                    {'+' if delta_circuit > 0 else ''}{delta_circuit:.3f}s
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="info-box">
                <div style="color: #AAAAAA; font-size: 0.8rem;">Driver Average</div>
                <div style="color: #FFFFFF; font-size: 1.1rem;">{format_time(driver_avg)}</div>
                <div style="color: {'#00FF00' if delta_driver < 0 else '#FF4444'}; font-size: 0.9rem;">
                    {'+' if delta_driver > 0 else ''}{delta_driver:.3f}s
                </div>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# TAB 2: MODEL INFO
# =============================================================================

def render_model_tab():
    """Render model information page."""

    model_info = get_model_info()

    if not model_info:
        st.warning("Cannot fetch model information. Is the API running?")
        return

    st.markdown("### Current Model")

    # Model identity
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Model Type</div>
            <div class="metric-value" style="font-size: 1.5rem;">
                {model_info.get('model_family', 'N/A').replace('_', ' ').title()}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Run Name</div>
            <div class="metric-value" style="font-size: 1.2rem;">
                {model_info.get('run_name', 'N/A')}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Source</div>
            <div class="metric-value" style="font-size: 1.5rem;">
                {model_info.get('source', 'N/A').upper()}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Performance Metrics")

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        mae = model_info.get('test_mae') or 0.0
        mae_display = f"{mae:.3f}s" if mae > 0 else "N/A"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Test MAE</div>
            <div class="metric-value">{mae_display}</div>
            <div style="color: #AAAAAA; font-size: 0.8rem;">Mean Absolute Error</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        r2 = model_info.get('test_r2') or 0.0
        r2_display = f"{r2:.3f}" if r2 > 0 else "N/A"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Test R²</div>
            <div class="metric-value">{r2_display}</div>
            <div style="color: #AAAAAA; font-size: 0.8rem;">Coefficient of Determination</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        cv_mae = model_info.get('cv_mae') or 0.0
        cv_mae_display = f"{cv_mae:.3f}s" if cv_mae > 0 else "N/A"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">CV MAE</div>
            <div class="metric-value">{cv_mae_display}</div>
            <div style="color: #AAAAAA; font-size: 0.8rem;">Cross-Validation</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        cv_r2 = model_info.get('cv_r2') or 0.0
        cv_r2_display = f"{cv_r2:.3f}" if cv_r2 > 0 else "N/A"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">CV R²</div>
            <div class="metric-value">{cv_r2_display}</div>
            <div style="color: #AAAAAA; font-size: 0.8rem;">Cross-Validation</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Additional info
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Model Details")
        st.markdown(f"""
        <div class="info-box">
            <p><strong>Run ID:</strong> <code>{model_info.get('run_id', 'N/A')}</code></p>
            <p><strong>Strategy:</strong> {model_info.get('strategy', 'N/A')}</p>
            <p><strong>RMSE:</strong> {model_info.get('test_rmse', 0):.3f}s</p>
            <p><strong>Overfitting Ratio:</strong> {model_info.get('overfitting_ratio', 0):.2f}</p>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("### Interpretation")
        mae = model_info.get('test_mae', 0)
        st.markdown(f"""
        <div class="info-box">
            <p>The model predicts lap times with an average error of <strong>{mae:.2f} seconds</strong>.</p>
            <p>This means predictions are typically within ±{mae:.1f}s of actual lap times.</p>
            <p style="color: #AAAAAA; font-size: 0.9rem;">
                Note: Professional F1 teams with telemetry data achieve ~0.1-0.2s MAE.
                This model uses only publicly available data.
            </p>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# TAB 3: LINKS
# =============================================================================

def render_links_tab():
    """Render links page."""

    st.markdown("### Project Resources")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <a href="{API_EXTERNAL_URL}/docs" target="_blank" style="text-decoration: none;">
            <div class="link-card">
                <h3 style="color: #E10600; margin: 0;">📡 FastAPI</h3>
                <p style="color: #AAAAAA;">REST API Documentation (Swagger)</p>
                <p style="color: #666666; font-size: 0.8rem;">{API_EXTERNAL_URL}/docs</p>
            </div>
        </a>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(f"""
        <a href="{MLFLOW_URL}" target="_blank" style="text-decoration: none;">
            <div class="link-card">
                <h3 style="color: #E10600; margin: 0;">📊 MLflow</h3>
                <p style="color: #AAAAAA;">ML Experiment Tracking</p>
                <p style="color: #666666; font-size: 0.8rem;">{MLFLOW_URL}</p>
            </div>
        </a>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <a href="{GITHUB_URL}" target="_blank" style="text-decoration: none;">
            <div class="link-card">
                <h3 style="color: #E10600; margin: 0;">💻 GitHub</h3>
                <p style="color: #AAAAAA;">Source Code Repository</p>
                <p style="color: #666666; font-size: 0.8rem;">{GITHUB_URL}</p>
            </div>
        </a>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="link-card" style="opacity: 0.5;">
            <h3 style="color: #666666; margin: 0;">📈 Evidently</h3>
            <p style="color: #666666;">Data Drift Monitoring (Coming Soon)</p>
            <p style="color: #444444; font-size: 0.8rem;">{EVIDENTLY_URL}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    # API Health check
    st.markdown("### System Status")

    health = api_request("/health")

    if health:
        col1, col2, col3 = st.columns(3)

        with col1:
            status = "🟢" if health.get("model_loaded") else "🔴"
            st.markdown(f"""
            <div class="info-box">
                <span style="font-size: 1.5rem;">{status}</span>
                <span style="color: #FFFFFF; margin-left: 0.5rem;">ML Model</span>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            status = "🟢" if health.get("database_connected") else "🔴"
            st.markdown(f"""
            <div class="info-box">
                <span style="font-size: 1.5rem;">{status}</span>
                <span style="color: #FFFFFF; margin-left: 0.5rem;">Database</span>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            status = "🟢" if health.get("mlflow_connected") else "🔴"
            st.markdown(f"""
            <div class="info-box">
                <span style="font-size: 1.5rem;">{status}</span>
                <span style="color: #FFFFFF; margin-left: 0.5rem;">MLflow</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.error("Cannot connect to API")


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    """Main application."""
    render_header()

    # Tabs
    tab1, tab2, tab3 = st.tabs(["🏎️ Prediction", "📊 Model", "🔗 Links"])

    with tab1:
        render_prediction_tab()

    with tab2:
        render_model_tab()

    with tab3:
        render_links_tab()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666666; font-size: 0.8rem; padding: 1rem;">
        F1PA - Formula 1 Predictive Assistant |
        <a href="https://github.com/Aurelien-L/F1PA" style="color: #E10600;">GitHub</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
