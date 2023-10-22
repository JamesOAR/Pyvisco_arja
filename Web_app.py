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

#Define Functions 


# Streamlit App
st.title("Pyvisco Web App")
# Option to choose the data source

data_source = st.radio("Select Raw data Source:", ("Upload Data", "Local Folder"))
domain_source=st.radio("Select domain of test data", ("Frequency", "Time"))
if domain_source=="Frequency":
    domain="freq"
else:
    domain="time"
module=st.radio("Select data type i.e. Tensile or Shear", ("Tensile", "Shear"))
if module=="Tensile":
    modul="E"
else:
    modul="G"

if data_source == "Upload Raw Data":
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        data = load.file(uploaded_file)
        df_raw,arr_RefT,units=load.user_raw(data,domain,modul)
        st.subheader("Uploaded Data")
        st.write(df_raw)
else:
    folder_path = st.text_input("Enter the path to the local file:")
    if folder_path:
        try:
            data=load.file(folder_path)
            df_raw,arr_RefT,units = load.user_raw(data,domain,modul)
            df_raw.modul=modul
            df_raw.domain=domain
            st.subheader("Raw data from Local file")
            st.write(df_raw)
        except Exception as e:
            st.error(f"Error: {str(e)}")
refT=st.select_slider("Reference Temperature", options=arr_RefT,value=25.0)
#Get shift factors and choose reference temperatue    
df_aT,dshift=master.get_aT(df_raw,refT) 
df_master=master.get_curve(df_raw,df_aT,refT)  

fig_master=master.plot(df_master,units)
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
