import streamlit as st

var=st.session_state

def lbs_to_kg():
    var.kg=var.lbs*0.4536
    
def kg_to_lbs():
    var.lbs=var.kg/0.4536

'object',var


col1,buff,col2 =st.columns([2,1,2])

with col1:
    pounds=st.number_input("weight in lbs",key='lbs',on_change=lbs_to_kg)
with col2:
    kilo=st.number_input( ' weight in kilo', key='kg',on_change=kg_to_lbs)
