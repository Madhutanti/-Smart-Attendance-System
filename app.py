import streamlit as st
import pandas as pd 
import time
from datetime import datetime

st.subheader("Attendance of the Candidate")
ts=time.time()
date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
timestamp=datetime.fromtimestamp(ts).strftime("%H:%M:%S")

df = pd.read_csv("Attendance/Attendance_"+date+".csv")
df.insert(0,"S.NO.",range(1,len(df)+1))



#st.dataframe(df.style.highlight_max(axis=0))
st.dataframe(df.style.applymap(lambda x:"background-color:yellow"))