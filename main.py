import streamlit as st

# File uploader widget
uploaded_file = st.file_uploader("Upload a file")

# Check if file is uploaded
if uploaded_file is not None:
    # Extract file name
    file_name = uploaded_file.name

    # Display file name
    st.write("File Name:", file_name)
