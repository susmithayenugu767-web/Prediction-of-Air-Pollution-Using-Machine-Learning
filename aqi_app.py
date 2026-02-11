# =========================
# IMPORT LIBRARIES
# =========================
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import datetime
from datetime import datetime as dt
import warnings
warnings.filterwarnings('ignore')
from PIL import Image
from streamlit_option_menu import option_menu

# =========================
# PAGE CONFIGURATION
# =========================
# Page icon (you can add your own logo)
try:
    icon = Image.open(r"c:\Users\susmi\Downloads\AQI.jpeg")  # Add your logo file
except:
    icon = None

# Page configuration
st.set_page_config(
    page_title="Prediction of Air Pollution Using Machine Learning",
    page_icon=icon,
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# CUSTOM CSS STYLING
# =========================
st.markdown("""
    <style>
    /* Header styling */
    .header-title {
        font-size: 35px;
        font-weight: medium;
        color: #000080;
        text-align: center;
        margin-bottom: 10px;
    }
    .subheader-title {
        font-size: 24px;
        font-weight: medium;
        color: #BDB76B;
        text-align: center;
        margin-bottom: 30px;
    }

    /* Card styling */
    .card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        background-color: white;
    }
    .card h3 {
        margin: 0;
        font-size: 18px;
        color: #333;
    }
    .card p {
        margin: 4px 0;
        font-size: 14px;
        color: #666;
    }

    /* Button styling */
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 10px 20px;
        cursor: pointer;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }

    /* Reset button styling */
    .reset-btn {
        background-color: #ff4444 !important;
        color: white !important;
    }
    .reset-btn:hover {
        background-color: #cc0000 !important;
    }

    /* Input field styling */
    .stNumberInput > div > div > input {
        border-radius: 4px;
        border: 1px solid #ddd;
    }

    /* Project title styling */
    .project-title {
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        color: #000080;
        margin-bottom: 10px;
        text-transform: uppercase;
    }
    .project-subtitle {
        font-size: 28px;
        font-weight: bold;
        text-align: center;
        color: #4CAF50;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# =========================
# HEADER SECTION
# =========================
st.title("Yenugu Susmitha Reddy")
st.subheader("Prediction of Air Pollution Using Machine Learning")


# =========================
# SIDEBAR NAVIGATION
# =========================
with st.sidebar:
    if icon:
        st.sidebar.image(icon, use_container_width=True)

    selected = option_menu(
        menu_title="🌍 Navigation",
        options=["Home", "AQI Prediction", "Historical Data", "City Analysis", "About"],
        icons=["house", "speedometer2", "clock-history", "building", "info-circle"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#f8f9fa"},
            "icon": {"color": "orange", "font-size": "18px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "5px",
                "--hover-color": "#e9ecef"
            },
            "nav-link-selected": {"background-color": "#4CAF50", "color": "white"},
        }
    )

# Add balloons effect for welcome
if selected == "Home":
    st.balloons()

# =========================
# DATA LOADING FUNCTIONS
# =========================
@st.cache_data
def load_model():
    try:
        with open("air_pollution_model.pkl", "rb") as file:
            return pickle.load(file)
    except:
        st.error("Model file not found. Please train the model first.")
        return None

@st.cache_data
def load_visualization_data():
    try:
        data = pd.read_csv("aqi_visualization_data.csv", parse_dates=['Date'])
        return data
    except:
        st.warning("Historical data not available. Run data processing script first.")
        return None

@st.cache_data
def load_pollutant_stats():
    try:
        with open('pollutant_statistics.json', 'r') as f:
            return json.load(f)
    except:
        return None

@st.cache_data
def load_example_scenarios():
    try:
        with open('example_scenarios.json', 'r') as f:
            scenarios = json.load(f)
            # Remove Typical Hyderabad Day and Typical Bangalore Day
            if 'Typical Hyderabad Day' in scenarios:
                del scenarios['Typical Hyderabad Day']
            if 'Typical Bangalore Day' in scenarios:
                del scenarios['Typical Bangalore Day']
            return scenarios
    except:
        return {
            'Clean Air Day': {'CO': 2.0, 'Ozone': 25.0, 'PM10': 15.0, 'PM25': 10.0, 'NO2': 15.0},
            'Moderate Pollution': {'CO': 5.0, 'Ozone': 50.0, 'PM10': 35.0, 'PM25': 25.0, 'NO2': 30.0},
            'High Pollution': {'CO': 10.0, 'Ozone': 80.0, 'PM10': 60.0, 'PM25': 50.0, 'NO2': 60.0}
        }

# =========================
# HOME PAGE
# =========================
if selected == "Home":
    st.title("🌤️ Welcome to Air Pollution Prediction System")
    st.markdown("---")

    # Introduction cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="card">
            <h3>🤖 ML-based Prediction</h3>
            <p>Predict Air Quality Index using advanced machine learning algorithms</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <h3>📈 Historical Analysis</h3>
            <p>Explore air pollution trends from 2020-2025 across multiple cities</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="card">
            <h3>🏙️ City Comparison</h3>
            <p>Compare air pollution levels between different Indian cities</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # System Overview
    st.subheader("📋 System Overview")
    st.markdown("""
    This system **"Prediction of Air Pollution Using Machine Learning"** is designed to monitor, analyze, 
    and forecast air quality levels using advanced machine learning techniques. The system leverages 
    historical air quality data to predict Air Quality Index (AQI) and provides actionable insights 
    for environmental management and public health protection.

    ### Key Features:
    - **Real-time AQI Prediction**: Input pollutant values to get instant AQI predictions
    - **Historical Data Analysis**: Explore 5 years of comprehensive air quality data
    - **City-wise Comparison**: Compare pollution levels across multiple cities
    - **Interactive Visualizations**: Dynamic charts and graphs for better understanding
    - **Educational Scenarios**: Learn about different pollution scenarios
    """)

    # How to Use
    st.markdown("---")
    st.subheader("🚀 How to Use This System")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 1. AQI Prediction
        - Go to **AQI Prediction** tab
        - Enter pollutant values manually
        - Or select from predefined scenarios
        - Click 'Predict AQI' to get results
        """)

    with col2:
        st.markdown("""
        ### 2. Historical Analysis
        - Navigate to **Historical Data** tab
        - Select date range and cities
        - Choose pollutant to analyze
        - Click 'Apply Filters & Analyze'
        """)

    with col3:
        st.markdown("""
        ### 3. City Analysis
        - Go to **City Analysis** tab
        - Select a city from dropdown
        - View detailed pollution statistics
        - Analyze trends and patterns
        """)

# =========================
# AQI PREDICTION PAGE
# =========================
elif selected == "AQI Prediction":
    st.title("📊 Air Pollution Prediction Dashboard")

    # Load model and data
    model = load_model()
    pollutant_stats = load_pollutant_stats()
    example_scenarios = load_example_scenarios()

    if model is None:
        st.error("Model not available. Please train the model first.")
        st.stop()

    # Create tabs like Price Tracker
    tab1, tab2, tab3 = st.tabs(["Manual Input", "Quick Scenarios", "Guidance"])

    with tab1:
        st.subheader("📝 Enter Pollutant Values")

        # Two-column layout
        col1, col2 = st.columns(2)

        with col1:
            # CO input - DIRECT NUMBER INPUT (not slider)
            st.write("**CO (Carbon Monoxide)**")
            st.write("Units: ppm (parts per million)")
            co = st.number_input(
                "Enter CO value:",
                min_value=0.0,
                max_value=100.0,
                value=5.0,
                step=0.1,
                key="co_input"
            )

            # Ozone input - DIRECT NUMBER INPUT
            st.write("**Ozone (O3)**")
            st.write("Units: ppb (parts per billion)")
            o3 = st.number_input(
                "Enter Ozone value:",
                min_value=0.0,
                max_value=300.0,
                value=30.0,
                step=0.5,
                key="o3_input"
            )

        with col2:
            # PM10 input - DIRECT NUMBER INPUT
            st.write("**PM10**")
            st.write("Units: μg/m³ (micrograms per cubic meter)")
            pm10 = st.number_input(
                "Enter PM10 value:",
                min_value=0.0,
                max_value=500.0,
                value=15.0,
                step=1.0,
                key="pm10_input"
            )

            # PM2.5 input - DIRECT NUMBER INPUT
            st.write("**PM2.5**")
            st.write("Units: μg/m³ (micrograms per cubic meter)")
            pm25 = st.number_input(
                "Enter PM2.5 value:",
                min_value=0.0,
                max_value=500.0,
                value=25.0,
                step=1.0,
                key="pm25_input"
            )

            # NO2 input - DIRECT NUMBER INPUT
            st.write("**NO2 (Nitrogen Dioxide)**")
            st.write("Units: ppb (parts per billion)")
            no2 = st.number_input(
                "Enter NO2 value:",
                min_value=0.0,
                max_value=200.0,
                value=20.0,
                step=0.5,
                key="no2_input"
            )

    with tab2:
        st.subheader("🎯 Quick Scenario Selection")

        if example_scenarios:
            scenario_names = list(example_scenarios.keys())
            selected_scenario = st.selectbox("Choose a scenario:", scenario_names)

            if selected_scenario:
                scenario = example_scenarios[selected_scenario]

                col1, col2 = st.columns([2, 1])
                with col1:
                    st.info(f"**{selected_scenario}**")
                    # Apply button
                    if st.button("Apply This Scenario", type="primary", key="apply_scenario"):
                        st.session_state.co = scenario.get('CO', 5.0)
                        st.session_state.o3 = scenario.get('Ozone', 30.0)
                        st.session_state.pm10 = scenario.get('PM10', 15.0)
                        st.session_state.pm25 = scenario.get('PM25', 25.0)
                        st.session_state.no2 = scenario.get('NO2', 20.0)
                        st.success("Scenario applied! Switch to Manual Input tab.")

                with col2:
                    # Show scenario values
                    scenario_df = pd.DataFrame({
                        'Pollutant': ['CO', 'Ozone', 'PM10', 'PM2.5', 'NO2'],
                        'Value': [
                            scenario.get('CO', 5.0), scenario.get('Ozone', 30.0),
                            scenario.get('PM10', 15.0), scenario.get('PM25', 25.0),
                            scenario.get('NO2', 20.0)
                        ]
                    })
                    st.dataframe(scenario_df, use_container_width=True)

    with tab3:
        st.subheader("ℹ️ Input Value Guidance")
        st.markdown("""
        ### Understanding Pollutant Units:

        **1. CO (Carbon Monoxide)** - Measured in ppm
        - **0-5 ppm**: Good air quality
        - **5-10 ppm**: Moderate pollution
        - **10+ ppm**: High pollution

        **2. Ozone (O3)** - Measured in ppb
        - **0-50 ppb**: Good
        - **50-100 ppb**: Moderate
        - **100+ ppb**: Unhealthy

        **3. PM10** - Particulate Matter ≤10μm (μg/m³)
        - **0-50 μg/m³**: Good
        - **50-100 μg/m³**: Moderate
        - **100+ μg/m³**: Unhealthy

        **4. PM2.5** - Fine Particulate Matter (μg/m³)
        - **0-25 μg/m³**: Good
        - **25-50 μg/m³**: Moderate
        - **50+ μg/m³**: Unhealthy

        **5. NO2** - Nitrogen Dioxide (ppb)
        - **0-40 ppb**: Good
        - **40-80 ppb**: Moderate
        - **80+ ppb**: Unhealthy
        """)

    # Prediction button and Reset button
    st.markdown("---")
    predict_col1, predict_col2, predict_col3 = st.columns([2, 1, 1])

    with predict_col1:
        st.markdown("### 🚀 Ready to Predict")

    with predict_col2:
        if st.button("Predict AQI", type="primary", use_container_width=True):
            # Get input values
            co_val = st.session_state.get('co', co)
            o3_val = st.session_state.get('o3', o3)
            pm10_val = st.session_state.get('pm10', pm10)
            pm25_val = st.session_state.get('pm25', pm25)
            no2_val = st.session_state.get('no2', no2)

            # Make prediction
            try:
                input_data = np.array([[co_val, o3_val, pm10_val, pm25_val, no2_val]])
                prediction = model.predict(input_data)
                aqi_value = int(prediction[0])

                # Store in session
                st.session_state.prediction = aqi_value
                st.session_state.show_result = True
            except Exception as e:
                st.error(f"Prediction error: {e}")

    with predict_col3:
        # RESET BUTTON for AQI Prediction page only
        if st.button("🔄 Reset", key="reset_prediction", use_container_width=True, 
                    type="secondary"):
            # Clear session state for this page
            keys_to_clear = ['co', 'o3', 'pm10', 'pm25', 'no2', 'prediction', 'show_result']
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("Inputs reset! Enter new values.")
            st.rerun()

    # Show prediction result
    if st.session_state.get('show_result', False):
        aqi_value = st.session_state.prediction

        # AQI Categories
        if aqi_value <= 50:
            category = "Good"
            color = "#00E400"
            icon = "😊"
            advice = "Air quality is satisfactory"
        elif aqi_value <= 100:
            category = "Moderate"
            color = "#FFFF00"
            icon = "😐"
            advice = "Acceptable air quality"
        elif aqi_value <= 150:
            category = "Unhealthy for Sensitive Groups"
            color = "#FF7E00"
            icon = "😷"
            advice = "Sensitive groups should take caution"
        elif aqi_value <= 200:
            category = "Unhealthy"
            color = "#FF0000"
            icon = "😟"
            advice = "Everyone may be affected"
        elif aqi_value <= 300:
            category = "Very Unhealthy"
            color = "#8F3F97"
            icon = "🚨"
            advice = "Health alert"
        else:
            category = "Hazardous"
            color = "#7E0023"
            icon = "⚠️"
            advice = "Emergency conditions"

        # Display result in card
        st.markdown(f"""
        <div class="card" style="border-left: 10px solid {color};">
            <div style="text-align: center;">
                <h1 style="color: {color}; margin: 0;">{icon} AQI: {aqi_value}</h1>
                <h3 style="color: {color}; margin: 10px 0;">{category}</h3>
                <p style="font-size: 16px;">{advice}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# =========================
# HISTORICAL DATA PAGE
# =========================

elif selected == "Historical Data":
    st.title("📊 Historical Data Explorer")

    viz_data = load_visualization_data()

    if viz_data is not None:
        # Initialize session state for filters if not exists
        if 'filters_applied' not in st.session_state:
            st.session_state.filters_applied = False

        # Get min and max dates from data
        min_date = viz_data['Date'].min().date()
        max_date = viz_data['Date'].max().date()

        # Use session state values or defaults
        from_date_value = st.session_state.get('from_date_value', min_date)
        to_date_value = st.session_state.get('to_date_value', max_date)

        # Filters section
        st.subheader("🔍 Filter Options")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            from_date = st.date_input("From:", value=from_date_value, 
                                     min_value=min_date, max_value=max_date,
                                     key="from_date_widget")

        with col2:
            to_date = st.date_input("To:", value=to_date_value,
                                   min_value=min_date, max_value=max_date,
                                   key="to_date_widget")

        with col3:
            if 'Site Name (of Overall AQI)' in viz_data.columns:
                cities = sorted(viz_data['Site Name (of Overall AQI)'].unique())
                # Get selected cities from session state or default to empty list
                default_cities = st.session_state.get('selected_cities_value', [])
                selected_cities = st.multiselect("Select Cities:", cities, 
                                                default=default_cities,
                                                key="cities_widget")

        with col4:
            pollutant_options = ['CO', 'Ozone', 'PM10', 'PM25', 'NO2', 'Overall AQI Value']
            default_pollutant = st.session_state.get('selected_pollutant_value', 'Overall AQI Value')
            selected_pollutant = st.selectbox("Select Pollutant:", pollutant_options, 
                                             index=pollutant_options.index(default_pollutant) if default_pollutant in pollutant_options else 0,
                                             key="pollutant_widget")

        # Apply button and Reset button
        st.markdown("---")
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            apply_pressed = st.button("Apply Filters & Analyze", type="primary", use_container_width=True)

        with col2:
            reset_pressed = st.button("🔄 Reset Filters", key="reset_history", 
                                     use_container_width=True, type="secondary")

        # Handle Apply button
        if apply_pressed:
            # Store filter values in session state (with different names to avoid conflict)
            st.session_state.from_date_value = from_date
            st.session_state.to_date_value = to_date
            st.session_state.selected_cities_value = selected_cities
            st.session_state.selected_pollutant_value = selected_pollutant
            st.session_state.filters_applied = True
            st.rerun()

        # Handle Reset button
        if reset_pressed:
            # Clear filter values from session state
            keys_to_clear = ['filters_applied', 'from_date_value', 'to_date_value', 
                            'selected_cities_value', 'selected_pollutant_value']
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("Filters reset!")
            st.rerun()

        # Apply filters if set
        filtered_data = viz_data.copy()

        if st.session_state.get('filters_applied', False):
            # Get values from session state
            from_date_val = st.session_state.get('from_date_value')
            to_date_val = st.session_state.get('to_date_value')
            selected_cities_val = st.session_state.get('selected_cities_value', [])
            selected_pollutant_val = st.session_state.get('selected_pollutant_value', 'Overall AQI Value')

            if from_date_val and to_date_val:
                filtered_data = filtered_data[
                    (filtered_data['Date'].dt.date >= from_date_val) & 
                    (filtered_data['Date'].dt.date <= to_date_val)
                ]

            if selected_cities_val:
                filtered_data = filtered_data[filtered_data['Site Name (of Overall AQI)'].isin(selected_cities_val)]

            # Display results
            st.markdown(f"**Showing:** {len(filtered_data):,} records")

            if len(filtered_data) == 0:
                st.warning("No data found with the selected filters. Try different filters.")
            else:
                # Tabs - Now 4 tabs with City-wise AQI added
                tab1, tab2, tab3, tab4 = st.tabs(["Time Trends", "City Comparison", "City-wise AQI", "Statistics"])

                with tab1:
                    st.subheader("📈 Time Series Analysis")

                    fig = px.line(filtered_data.sort_values('Date'), x='Date', y=selected_pollutant_val,
                                color='Site Name (of Overall AQI)' if 'Site Name (of Overall AQI)' in filtered_data.columns else None,
                                title=f'{selected_pollutant_val} Over Time')
                    st.plotly_chart(fig, use_container_width=True)

                with tab2:
                    st.subheader("🌍 City Comparison")

                    if 'Site Name (of Overall AQI)' in filtered_data.columns:
                        city_stats = filtered_data.groupby('Site Name (of Overall AQI)')[selected_pollutant_val].agg(['mean', 'min', 'max']).round(2)
                        st.dataframe(city_stats, use_container_width=True)

                        fig = px.bar(city_stats.reset_index(), x='Site Name (of Overall AQI)', y='mean',
                                    title=f'Average {selected_pollutant_val} by City')
                        st.plotly_chart(fig, use_container_width=True)

                with tab3:
                    st.subheader("🏙️ City-wise AQI Distribution")

                    # City-wise AQI average
                    city_avg = filtered_data.groupby('Site Name (of Overall AQI)')['Overall AQI Value'].mean().reset_index()

                    # Create bar chart
                    fig = px.bar(city_avg, x='Site Name (of Overall AQI)', y='Overall AQI Value',
                                title='Average AQI by City', 
                                color='Site Name (of Overall AQI)',
                                text='Overall AQI Value',
                                color_discrete_sequence=px.colors.qualitative.Set2)
                    fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
                    fig.update_layout(xaxis_title="City", yaxis_title="Average AQI")
                    st.plotly_chart(fig, use_container_width=True)

                    # Display city statistics
                    st.subheader("City-wise AQI Statistics")
                    col1, col2, col3, col4 = st.columns(4)

                    for idx, city in enumerate(city_avg['Site Name (of Overall AQI)'].unique()):
                        city_data = filtered_data[filtered_data['Site Name (of Overall AQI)'] == city]
                        with col1 if idx % 4 == 0 else col2 if idx % 4 == 1 else col3 if idx % 4 == 2 else col4:
                            avg_aqi = city_data['Overall AQI Value'].mean()
                            st.metric(f"{city} AQI", f"{avg_aqi:.1f}")

                with tab4:
                    st.subheader("📊 Statistical Analysis")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Average", f"{filtered_data[selected_pollutant_val].mean():.2f}")
                    with col2:
                        st.metric("Median", f"{filtered_data[selected_pollutant_val].median():.2f}")
                    with col3:
                        st.metric("Minimum", f"{filtered_data[selected_pollutant_val].min():.2f}")
                    with col4:
                        st.metric("Maximum", f"{filtered_data[selected_pollutant_val].max():.2f}")

                    fig = px.histogram(filtered_data, x=selected_pollutant_val, title='Distribution')
                    st.plotly_chart(fig, use_container_width=True)
        else:
            if not reset_pressed:  # Don't show this message when resetting
                st.info("👆 Please select filters and click 'Apply Filters & Analyze' to see the data.")

    else:
        st.error("Historical data not available. Please run the data processing script first.")


# =========================
# CITY ANALYSIS PAGE
# =========================
elif selected == "City Analysis":
    st.title("🏙️ City-wise Air Pollution Analysis")

    viz_data = load_visualization_data()

    if viz_data is not None:
        # City selection with Reset button
        col1, col2 = st.columns([3, 1])

        with col1:
            cities = sorted(viz_data['Site Name (of Overall AQI)'].unique())
            selected_city = st.selectbox("Select a City:", cities, key="city_select")

        with col2:
            # RESET BUTTON for City Analysis page
            if st.button("🔄 Reset", key="reset_city", use_container_width=True, 
                        type="secondary"):
                keys_to_clear = ['city_select']
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                st.success("City selection reset!")
                st.rerun()

        if selected_city:
            city_data = viz_data[viz_data['Site Name (of Overall AQI)'] == selected_city]

            # City metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Average AQI", f"{city_data['Overall AQI Value'].mean():.1f}")
            with col2:
                st.metric("Best AQI", f"{city_data['Overall AQI Value'].min():.1f}")
            with col3:
                st.metric("Worst AQI", f"{city_data['Overall AQI Value'].max():.1f}")
            with col4:
                st.metric("Records", f"{len(city_data):,}")

            # Time series for selected city
            fig = px.line(city_data.sort_values('Date'), x='Date', y='Overall AQI Value',
                         title=f'AQI Trend in {selected_city}')
            st.plotly_chart(fig, use_container_width=True)

            # Monthly patterns
            city_data['Month-Year'] = city_data['Date'].dt.strftime('%b %Y')
            monthly_avg = city_data.groupby('Month-Year')['Overall AQI Value'].mean().reset_index()

            fig2 = px.bar(monthly_avg, x='Month-Year', y='Overall AQI Value',
                         title=f'Monthly Average AQI in {selected_city}')
            st.plotly_chart(fig2, use_container_width=True)

            # Pollutant analysis for the city
            st.subheader("📊 Pollutant Analysis")
            pollutant_cols = ['CO', 'Ozone', 'PM10', 'PM25', 'NO2']

            pollutant_avg = city_data[pollutant_cols].mean().reset_index()
            pollutant_avg.columns = ['Pollutant', 'Average']

            fig3 = px.bar(pollutant_avg, x='Pollutant', y='Average',
                         title=f'Average Pollutant Levels in {selected_city}')
            st.plotly_chart(fig3, use_container_width=True)

# =========================
# ABOUT PAGE
# =========================
elif selected == "About":
    st.title("About Air Pollution Prediction System")

    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("""
        ## 🌍 Project Overview

        This comprehensive system **"Prediction of Air Pollution Using Machine Learning"** is designed to 
        monitor, analyze, and forecast air quality levels using advanced machine learning techniques. 
        The system leverages historical air quality data to predict Air Quality Index (AQI) and provides 
        actionable insights for environmental management and public health protection.
        """)

    with col2:
        # Add project logo if available
        try:
            logo = Image.open(r"c:\Users\susmi\Downloads\AQI.jpeg")
            st.image(logo, width=150)
        except:
            st.info("Project Logo")

    st.markdown("---")

    # Project Details in Tabs
    tab1, tab2, tab3,  = st.tabs(["🎯 Objectives", "🛠️ Methodology", "📊 Data & Model", ])

    with tab1:
        st.subheader("Project Objectives")
        st.markdown("""
        ### Primary Goals of the Project:

        1. **Air Pollution Prediction**: Develop an accurate machine learning model to predict 
           Air Quality Index (AQI) based on multiple pollutant parameters

        2. **Historical Trend Analysis**: Analyze 5 years of air quality data (2020-2025) 
           to identify pollution patterns and seasonal variations

        3. **Multi-city Comparison**: Enable comparative analysis of air pollution levels 
           across different Indian cities

        4. **Real-time Assessment**: Provide instant AQI predictions based on user-input 
           pollutant concentrations

        5. **Environmental Awareness**: Create an educational platform to increase public 
           awareness about air pollution and its health impacts

        """)

    with tab2:
        st.subheader("Methodology & Technical Approach")
        st.markdown("""
        ### 🧪 Implementation Methodology:

        **1. Data Collection & Preprocessing**
        - Collected comprehensive air quality data from monitoring stations
        - Handled missing values and data inconsistencies
        - Normalized pollutant measurements for machine learning compatibility
        - Created temporal features for time-series analysis

        **2. Machine Learning Implementation**
        - Selected Random Forest Regressor for its robustness in regression tasks
        - Used 5 key air pollutants as predictive features
        - Implemented cross-validation to ensure model generalizability
        - Optimized hyperparameters for maximum prediction accuracy

        **3. System Architecture**
        - Backend: Python-based machine learning pipeline
        - Frontend: Streamlit web application framework
        - Database: Processed CSV files with historical air quality data
        - Visualization: Interactive plots using Plotly and Matplotlib

        **4. Model Deployment**
        - Serialized trained model using Pickle
        - Created RESTful prediction endpoints
        - Implemented user-friendly interface with real-time feedback
        - Added scenario simulation for educational purposes
        """)

    with tab3:
        st.subheader("Data & Model Specifications")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### 📈 Air Quality Dataset:

            **Source**: Real-time monitoring stations across India

            **Temporal Coverage**: January 2020 - December 2025

            **Geographical Coverage**:
            - Hyderabad (Telangana)
            - Bangalore (Karnataka)
            - Delhi (National Capital Region)
            - Visakhapatnam (Andhra Pradesh)

            **Pollutant Parameters**:
            1. **CO (Carbon Monoxide)** - Measured in ppm (parts per million)
            2. **Ozone (O₃)** - Measured in ppb (parts per billion)
            3. **PM10** - Particulate Matter ≤10μm (μg/m³)
            4. **PM2.5** - Fine Particulate Matter ≤2.5μm (μg/m³)
            5. **NO₂** - Nitrogen Dioxide (ppb)

            **Target Variable**: Overall AQI Value
            """)

        with col2:
            st.markdown("""
            ### 🤖 Machine Learning Model:

            **Algorithm**: Random Forest Regressor

            **Input Features**: 5 pollutant concentrations

            **Output**: Predicted AQI Value

            **Model Performance Metrics**:
            - R² Score (Coefficient of Determination): **> 0.85**
            - Mean Absolute Error (MAE): **< 15 AQI points**
            - Root Mean Square Error (RMSE): **< 20 AQI points**

            **Data Split**:
            - Training Data: **80%** (Model development)
            - Testing Data: **20%** (Performance evaluation)

            **Feature Importance**:
            - PM2.5 and PM10 identified as most significant predictors
            - All 5 pollutants contribute to AQI prediction
            """)




    # Key Features Section
    st.subheader("✨ System Features & Capabilities")

    features = [
        {"icon": "🤖", "title": "ML-based Prediction", "desc": "Accurate AQI prediction using Random Forest algorithm"},
        {"icon": "📊", "title": "Real-time Analysis", "desc": "Instant AQI calculation based on pollutant inputs"},
        {"icon": "📈", "title": "Historical Data Explorer", "desc": "5-year comprehensive air quality data analysis"},
        {"icon": "🏙️", "title": "City Comparison", "desc": "Compare pollution levels across 4 major Indian cities"},
        {"icon": "🌫️", "title": "Pollutant Contribution", "desc": "Analyze individual pollutant impact on AQI"},
        {"icon": "🎯", "title": "Scenario Simulation", "desc": "Test various pollution scenarios and their AQI impact"},
        {"icon": "📱", "title": "User-friendly Interface", "desc": "Intuitive design with easy navigation"},
        {"icon": "📋", "title": "Comprehensive Reports", "desc": "Detailed statistical analysis and visualizations"}
    ]

    # Display features in 4 columns
    cols = st.columns(4)
    for i, feature in enumerate(features):
        with cols[i % 4]:
            st.markdown(f"""
            <div class="card" style="height: 200px; margin-bottom: 15px;">
                <div style="font-size: 28px; margin-bottom: 10px; text-align: center;">{feature['icon']}</div>
                <h4 style="margin: 5px 0; text-align: center; color: #000080;">{feature['title']}</h4>
                <p style="font-size: 13px; color: #666; text-align: center;">{feature['desc']}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Technical Stack
    st.subheader("🛠️ Technology Stack Used")

    st.markdown("""
    - **Programming:** Python 3.11 – Core development language  
    - **ML Framework:** Scikit-learn – Machine learning algorithms  
    - **Web Framework:** Streamlit – Interactive web application  
    - **Data Processing:** Pandas, NumPy – Data manipulation and analysis  
    - **Visualization:** Plotly, Matplotlib – Data plotting and charts  
    - **Model Storage:** Pickle – Model serialization  
    - **Data Storage:** JSON, CSV – Configuration and data files  
    - **Development:** Jupyter Notebook, VS Code – Development environment  
    """)



    st.markdown("---")

    # Applications & Impact
    st.subheader("📋 Practical Applications & Impact")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🏢 Real-world Applications:

        **Environmental Monitoring**
        - Continuous air quality tracking
        - Pollution source identification
        - Environmental compliance monitoring

        **Public Health Protection**
        - Daily air quality advisories
        - Sensitive group alerts (asthma, elderly)
        - Outdoor activity recommendations

        **Educational Tool**
        - Environmental science education
        - Public awareness campaigns
        - Research and academic projects

        """)

    with col2:
        st.markdown("""
        ### 🌱 Environmental & Social Impact:

        **Awareness Generation**
        - Educates public about air pollution dangers
        - Promotes environmental consciousness
        - Encourages sustainable practices

        **Health Benefits**
        - Helps prevent respiratory diseases
        - Reduces healthcare burden
        - Improves quality of life

        **Economic Impact**
        - Supports tourism industry
        - Attracts clean industry investments
        - Reduces pollution-related economic losses
        """)

    st.markdown("---")



    # Final Footer
    st.subheader("Prediction of Air Pollution Using Machine Learning")


# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("Prediction of Air Pollution Using Machine Learning ")
