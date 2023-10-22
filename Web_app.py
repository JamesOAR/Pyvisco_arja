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
    #df_raw, units = prep_csv(data)
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

#Choose the type of data you will be working on 
tab1,tab2,=st.tabs(["Raw Data","Master Curve"])

with tab1:
    st.header("Raw data Calibration")
# Option to choose the data source
    data_source = st.sidebar.radio("Select Raw data Source:", ( "Local Folder","Upload Data"))

    if data_source == "Upload Data":
        domain_source = st.sidebar.radio("Select domain of test data", ("Frequency", "Time"))
        module = st.sidebar.radio("Select data type i.e. Tensile or Shear", ("Tensile", "Shear"))

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
                #bytes_data = uploaded_file.read()
                data = pd.read_csv(uploaded_file)
                df_raw, arr_RefT, units=pull_data(data,domain,modul)
                #df_raw, arr_RefT, units = load.user_raw(data, domain, modul)
                st.subheader("Uploaded Data")
                st.write(df_raw)
            except Exception as e:
                st.error(f"Error: {str(e)}")

    elif data_source == "Local Folder":
        folder_path = st.text_input("Enter the path to the local file:")
        module = st.sidebar.radio("Select data type i.e. Tensile or Shear", ("Tensile", "Shear"))
        domain_source=st.sidebar.radio("Select domain of test data",("Frequency", "Time"))
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

    st.subheader("Shift Factors")
    mshift = st.checkbox("Extract Shift Factors")

    if mshift:
        refT = st.text_input("Enter the reference temperature", value=25)
        try:
            refT = float(refT)
            df_aT, dshift = master.get_aT(df_raw, refT)
            df_master = master.get_curve(df_raw, df_aT, refT)
            fig_master = master.plot(df_master, units)
        except ValueError:
            st.warning("Please enter a valid reference temperature as a number.")
    #st.pyplot(fig_master)
    #st.subheader("Fit shift factors with WLF & Polynomial functions" )

    st.subheader("Plot Shifted Master Curve")
    show_master=st.checkbox("Plot Shift data")
    if show_master:
        try:
            fig_master_shift,ax=master.plot_shift(df_raw,df_master,units)
            st.pyplot(fig_master_shift)
        except Exception as e:
                st.error(f"Error: {str(e)}")


    show_fits=st.checkbox("Show WLF & Polynomial Shift Factor fits")  
    if show_fits:
        st.subheader("Show WLF & Polynomial Shift Factor fits")
        try:
            df_C,df_K=shift.fit_poly(df_aT)
            df_WLF=shift.fit_WLF(refT,df_aT)
            fig_shift,df_shift=shift.plot(df_aT,df_WLF,df_C)
            st.pyplot(fig_shift)  
        except Exception as e:
                st.error(f"Error: {str(e)}")  


    smoothen=st.checkbox("Smoothen Master Curve data")
    if smoothen:
        window=st.slider("Median filter window size",min_value=0,max_value=100,value=1,step=1)
        st.subheader("Smoothen Master Curve")
        try:
            smooth_master=master.smooth(df_master,window)
            smooth_fig=master.plot_smooth(smooth_master,units)
            st.pyplot(smooth_fig)
        except Exception as e:
            st.error(f"Error:{str(e)}")
    else:   
        try:
            smooth_master=master.smooth(df_master,1)
            smooth_fig=master.plot_smooth(smooth_master,units)
            st.pyplot(smooth_fig)
        except Exception as e:
            st.error(f"Error:{str(e)}")
        

    window=st.radio("Select the type of local discretization",("exact","round","min"))
    n_prony=st.text_input("Number of Prony series to fit",value=0)
    df_dis=prony.discretize(smooth_master,window, nprony=int(n_prony))
    
    discreet=st.checkbox("Plot discretized master curve")
    if discreet:
        st.subheader("Plot Discretized Master Curve")
        try:
            shifted_master=prony.plot_dis(smooth_master,df_dis,units)
            #smooth_fig=master.plot_smooth(smooth_master,units)
            st.pyplot(shifted_master)
        except Exception as e:
            st.error(f"Error:{str(e)}")
            
    prony_series,df_GMaxw=prony.fit(df_dis,df_master=smooth_master,opt=False)
    prony_check=st.checkbox("Plot Generalized Maxwell Model Fit")
    st.write(prony_series)
    if prony_check:
        try:  
            prony_fit=prony.plot_fit(smooth_master,df_GMaxw,units)
            prony_para=prony.plot_param(prony_series,labels=["tau_i","alpha_i"])
            st.pyplot(prony_para)
        except Exception as e:
            st.error(f"Error:{str(e)}")
