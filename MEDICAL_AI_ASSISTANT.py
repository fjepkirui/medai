import streamlit as st


def main():
    """
    Main function for the Medical AI Assistant Streamlit application.

    This function sets up the page configuration, displays the main page content
    including title, welcome message, and information about available tools.
    It also includes an expandable section with information about the application's
    technology stack.

    Returns:
        None
    """

    # Set page configuration
    st.set_page_config(page_title="Medical AI Assistant", page_icon="🏥", layout="wide")

    # Main page content
    st.title("Medical AI Assistant")
    st.markdown(
        """
    ### Welcome to the Medical AI Assistant

    This application provides AI-powered tools to assist medical practitioners with:

    1. **EHR Summarization** - Summarize electronic health records
    2. **Clinical Dictation** - Convert speech to text for clinical notes
    3. **Clinical Letter Generation** - Generate clinical letters
    4. **Medical Knowledge Bot** - Get answers to medical questions

    Please use the sidebar to navigate to the different tools.
    """
    )

    # Information about the technologies used
    with st.expander("About this application"):
        st.markdown(
            """
        This application is built using:
        - Streamlit for the web interface
        - Llama3.2:3b model running on a local Ollama server
        - Python for backend processing

        It provides a user-friendly interface for medical practitioners to leverage AI for common tasks.
        """
        )


if __name__ == "__main__":
    main()
