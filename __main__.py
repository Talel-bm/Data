import app
import streamlit as st
def main():
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = 'create_test_data'

    # Navigation
    if st.session_state.page == 'create_test_data':
        app.create_test_data()
        if 'test_data' in st.session_state:
            if st.button("Next: Process Model"):
                st.session_state.page = 'process_model'
    elif st.session_state.page == 'process_model':
        app.process_model()

if __name__ == "__main__":
    main()
