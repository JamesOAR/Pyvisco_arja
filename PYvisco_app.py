# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 23:05:39 2024

@author: jamoo
"""

import streamlit as st 
from pyvisco import load
from pyvisco import shift
from pyvisco import master
from pyvisco import prony
from pyvisco import opt
from pyvisco import styles
from pyvisco import out
import pandas as pd
import numpy as np
import io
from scipy.optimize import curve_fit
import zipfile
import matplotlib.pyplot as plt
import io

#Define Functions 
def pull_data(data,domain,modul):
    df = pd.read_csv(io.BytesIO(data), header=[0,1])
    df.dropna(inplace=True)
    df.rename(columns=lambda s: s.strip(), inplace=True)
    units = dict(zip(df.columns.get_level_values(0).values, 
                     df.columns.get_level_values(1).values))
    df.columns = df.columns.droplevel(1)
    df_raw=df

    if domain == 'freq':
        df_raw.rename(columns={"f":"f_set", 
            "{}_stor".format(modul):'{}_stor'.format(modul), 
            "{}_loss".format(modul):'{}_loss'.format(modul), 
            "T":"T", 'Set':'Set'}, 
            inplace=True, errors='raise')
    elif domain == 'time':
        df_raw.rename(columns={"t":"t_set", 
            "{}_relax".format(modul):'{}_relax'.format(modul), 
            "T":"T", 'Set':'Set'}, inplace=True, errors='raise')
        df_raw['f_set'] = 1/df_raw['t_set']

    load.check_units(units, modul)
    units = load.get_units(units, modul, domain)

    arr_RefT = df_raw.groupby('Set')['T'].mean().round()
    df_raw = df_raw.join(arr_RefT, on='Set', rsuffix='_round')
    df_raw.domain = domain
    df_raw.modul = modul
    return df_raw, arr_RefT, units

st.title("Pyvisco Web App")
st.sidebar.write("Pyvisco")
data_source = st.sidebar.radio("Select Raw data Source:", ("Local Folder","Upload Data"))
domain_source = st.sidebar.radio("Select domain of test data", ("Frequency", "Time"))
module = st.sidebar.radio("Select data type i.e. Tensile or Shear", ("Tensile", "Shear"))

tab1, tab2 = st.tabs(["Raw Data", "Master Curve"])

with tab1:
    st.header("Master Curve Calibration")
    
    if data_source == "Upload Data":
        uploaded_file = st.file_uploader("Upload a CSV file", accept_multiple_files=False)

        if domain_source == "Frequency":
            domain = "freq"
        else:
            domain = "time"

        if module == "Tensile":
            modul = "E"
        else:
            modul = "G"

        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                df_raw, arr_RefT, units = pull_data(data, domain, modul)
                st.subheader("Uploaded Data")
                st.write(df_raw)
            except Exception as e:
                st.error(f"Error: {str(e)}")

    elif data_source == "Local Folder":
        folder_path = st.text_input("Enter the path to the local file:")
    
        if domain_source == "Frequency":
            domain = "freq"
        else:
            domain = "time"

        if module == "Tensile":
            modul = "E"
        else:
            modul = "G"

        if folder_path:
            try:
                data = load.file(folder_path)
                df_raw, arr_RefT, units = load.user_raw(data, domain, modul)
                df_raw.modul = modul
                df_raw.domain = domain
                st.subheader("Raw data from Local file")
                st.write(df_raw)
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    st.subheader("Plot Shifted Master Curve")
    show_master = st.checkbox("Plot Shift data")
    if show_master:
        try:
            fig_master_shift, ax = master.plot_shift(df_raw, df_master, units)
            st.pyplot(fig_master_shift)
        except Exception as e:
            st.error(f"Error: {str(e)}")

    show_fits = st.checkbox("Show WLF & Polynomial Shift Factor fits")  
    if show_fits:
        st.subheader("Show WLF & Polynomial Shift Factor fits")
        try:
            df_C, df_K = shift.fit_poly(df_aT)
            df_WLF = shift.fit_WLF(refT, df_aT)
            fig_shift, df_shift = shift.plot(df_aT, df_WLF, df_C)
            st.pyplot(fig_shift)  
        except Exception as e:
            st.error(f"Error: {str(e)}")  

with tab2:
    st.header("Master Curve Calibration")
    
    if data_source == "Upload Data":
        RefT = st.text_input("Enter the reference temperature", value=25)
        uploaded_master_file = st.file_uploader("Upload the master file", accept_multiple_files=False)
        uploaded_shift_factors = st.file_uploader("Upload the shift factors file", accept_multiple_files=False)
        
        if domain_source == "Frequency":
            domain = "freq"
        else:
            domain = "time"

        if module == "Tensile":
            modul = "E"
        else:
            modul = "G"

        if uploaded_master_file is not None:
            try:
                df_master, units = load.user_master(uploaded_master_file, domain, RefT, modul)
                df_aT = load.user_shift(uploaded_shift_factors)
                st.subheader("Uploaded Data")
                st.write(df_master, df_aT)
            except Exception as e:
                st.error(f"Error: {str(e)}")

    elif data_source == "Local Folder":
        folder_master_path = st.text_input("Enter the path to the master csv file:")
        folder_shift_path = st.text_input("Enter the path to the shift csv file:")
        RefT = st.text_input("Enter the reference temperature", value=25)
        
        if domain_source == "Frequency":
            domain = "freq"
        else:
            domain = "time"

        if module == "Tensile":
            modul = "E"
        else:
            modul = "G"

        if folder_master_path:
            try:
                data_master = load.file(folder_master_path)
                df, units = load.prep_csv(data_master)
                df['omega'] = 2*np.pi*df['f']
                df['t'] = 1/df['f'] 
                df_master = df
                df_master.modul = modul
                df_master.domain = domain
                
                data_aT = load.file(folder_shift_path)
                df_aT = load.user_shift(data_aT)
                st.subheader("Raw data from Local file")
                st.write(df_master, df_aT)
            except Exception as e:
                st.error(f"Error: {str(e)}")
                
    st.subheader("Plot Master Curve")
    show_master_curve = st.checkbox("Plot Master Curve data")
    if show_master_curve:
        try:
            fig_master = master.plot(df_master, units)
            st.pyplot(fig_master)
        except Exception as e:
            st.error(f"Error: {str(e)}")
            
    show_shift = st.checkbox("Show WLF & Polynomial Shift Factor fit")  
    if show_shift:
        st.subheader("Show WLF & Polynomial Shift Factors")
        try:
            df_C, df_K = shift.fit_poly(df_aT)
            st.write(df_C, df_K, df_aT)
            df_WLF = shift.fit_WLF(RefT, df_aT)
            st.write(df_WLF)
            fig_shift, df_shift = shift.plot(df_aT, df_WLF, df_C)
            st.pyplot(fig_shift)  
        except Exception as e:
            st.error(f"Error: {str(e)}")
            
    # Add Prony Series Fitting Section
    st.subheader("Prony Series Fitting")
    fit_prony = st.checkbox("Fit Prony Series")
    
    if fit_prony and 'df_master' in locals():
        try:
            # Create discrete spectrum for fitting
            df_dis = prony.discrete_spectrum(df_master)
            
            # Optional smoothing of master curve
            smooth_master = master.smooth(df_master) if st.checkbox("Smooth master curve before fitting") else df_master
            
            # Fit Prony series
            prony_series, df_GMaxw = prony.fit(df_dis, df_master=smooth_master, opt=False)
            
            # Display Prony series parameters
            st.subheader("Prony Series Parameters")
            st.write(prony_series)
            
            # Plot fitting results
            prony_check = st.checkbox("Plot Generalized Maxwell Model Fit")
            if prony_check:
                try:
                    col1, col2 = st.columns(2)
                    with col1:
                        prony_fit = prony.plot_fit(smooth_master, df_GMaxw, units)
                        st.pyplot(prony_fit)
                    with col2:
                        prony_para = prony.plot_param(prony_series, labels=["tau_i", "alpha_i"])
                        st.pyplot(prony_para)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    
        except Exception as e:
            st.error(f"Error in Prony series fitting: {str(e)}")
    elif fit_prony:
        st.warning("Please load master curve data first to perform Prony series fitting.")