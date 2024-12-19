# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 13:23:01 2024

@author: jamoo
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyvisco import load, shift, master, prony, out
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="PyVisco Fitting Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 16px;
    }
    .plot-container {
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Helper functions
@st.cache_data
def pull_data(data, domain, modul):
    df = pd.read_csv(io.BytesIO(data), header=[0,1])
    df.dropna(inplace=True)
    df.rename(columns=lambda s: s.strip(), inplace=True)
    units = dict(zip(df.columns.get_level_values(0).values, 
                     df.columns.get_level_values(1).values))
    df.columns = df.columns.droplevel(1)
    df_raw = df

    if domain == 'freq':
        df_raw.rename(columns={
            "f": "f_set", 
            f"{modul}_stor": f'{modul}_stor', 
            f"{modul}_loss": f'{modul}_loss', 
            "T": "T", 
            'Set': 'Set'
        }, inplace=True, errors='raise')
    elif domain == 'time':
        df_raw.rename(columns={
            "t": "t_set", 
            f"{modul}_relax": f'{modul}_relax', 
            "T": "T", 
            'Set': 'Set'
        }, inplace=True, errors='raise')
        df_raw['f_set'] = 1/df_raw['t_set']

    load.check_units(units, modul)
    units = load.get_units(units, modul, domain)

    arr_RefT = df_raw.groupby('Set')['T'].mean().round()
    df_raw = df_raw.join(arr_RefT, on='Set', rsuffix='_round')
    df_raw.domain = domain
    df_raw.modul = modul
    return df_raw, arr_RefT, units

def create_interactive_plot(df, x_col, y_cols, title, x_label, y_label):
    fig = go.Figure()
    for y_col in y_cols:
        fig.add_trace(go.Scatter(
            x=df[x_col],
            y=df[y_col],
            name=y_col,
            mode='lines+markers'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        template='plotly_white',
        showlegend=True,
        width=800,
        height=500
    )
    return fig

# Main App
def main():
    st.title("📊 PyVisco Analysis Dashboard")
    st.markdown("---")

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        data_source = st.radio("Select Data Source:", ("Upload Data", "Local Folder"))
        domain_source = st.radio("Domain:", ("Frequency", "Time"))
        module = st.radio("Data Type:", ("Tensile", "Shear"))
        
        domain = "freq" if domain_source == "Frequency" else "time"
        modul = "E" if module == "Tensile" else "G"

    # Main content tabs
    tab1, tab2 = st.tabs(["📈 Raw Data Analysis", "🔄 Master Curve Analysis"])

    with tab1:
        st.header("Raw Data Processing")
        
        # Data loading section
        with st.expander("1️⃣ Load Data", expanded=True):
            if data_source == "Upload Data":
                uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
                if uploaded_file:
                    try:
                        data = uploaded_file.read()
                        df_raw, arr_RefT, units = pull_data(data, domain, modul)
                        st.session_state['df_raw'] = df_raw
                        st.session_state['units'] = units
                        st.success("Data loaded successfully!")
                        
                        with st.expander("View Raw Data"):
                            st.dataframe(df_raw)
                    except Exception as e:
                        st.error(f"Error loading data: {str(e)}")
            else:
                folder_path = st.text_input("Enter local file path:")
                if folder_path:
                    try:
                        data = load.file(folder_path)
                        df_raw, arr_RefT, units = load.user_raw(data, domain, modul)
                        st.session_state['df_raw'] = df_raw
                        st.session_state['units'] = units
                        st.success("Data loaded successfully!")
                        
                        with st.expander("View Raw Data"):
                            st.dataframe(df_raw)
                    except Exception as e:
                        st.error(f"Error loading data: {str(e)}")

        # Shift factors section
        if 'df_raw' in st.session_state:
            with st.expander("2️⃣ Calculate Shift Factors", expanded=True):
                refT = st.number_input("Reference Temperature", value=25.0)
                if st.button("Calculate Shift Factors"):
                    with st.spinner("Calculating shift factors..."):
                        df_aT, dshift = master.get_aT(st.session_state['df_raw'], refT)
                        df_master = master.get_curve(st.session_state['df_raw'], df_aT, refT)
                        st.session_state['df_master'] = df_master
                        st.session_state['df_aT'] = df_aT
                        
                        # Plot shift factors
                        fig_shift, ax = master.plot_shift(st.session_state['df_raw'], 
                                                        df_master, 
                                                        st.session_state['units'])
                        st.pyplot(fig_shift)

            # Master curve smoothing
            if 'df_master' in st.session_state:
                with st.expander("3️⃣ Smooth Master Curve", expanded=True):
                    window = st.slider("Smoothing Window Size", 1, 100, 1)
                    smooth_master = master.smooth(st.session_state['df_master'], window)
                    st.session_state['smooth_master'] = smooth_master
                    
                    fig_smooth = master.plot_smooth(smooth_master, st.session_state['units'])
                    st.pyplot(fig_smooth)

    with tab2:
        if 'smooth_master' in st.session_state:
            st.header("Master Curve Analysis")
            
            # Prony series fitting
            with st.expander("🎯 Fit Prony Series", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    window_type = st.selectbox("Discretization Type", 
                                             ["exact", "round", "min"])
                with col2:
                    n_prony = st.number_input("Number of Prony Terms", 
                                            min_value=0, value=0)
                
                if st.button("Fit Prony Series"):
                    with st.spinner("Fitting Prony series..."):
                        df_dis = prony.discretize(st.session_state['smooth_master'], 
                                                window_type, 
                                                int(n_prony))
                        prony_series, df_GMaxw = prony.fit(df_dis, 
                                                          df_master=st.session_state['smooth_master'], 
                                                          opt=False)
                        
                        # Plot results
                        fig_prony = prony.plot_fit(st.session_state['smooth_master'], 
                                                 df_GMaxw, 
                                                 st.session_state['units'])
                        st.pyplot(fig_prony)
                        
                        # Display Prony parameters
                        st.subheader("Prony Series Parameters")
                        st.dataframe(prony_series['df_terms'])

if __name__ == "__main__":
    main()