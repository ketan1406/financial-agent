# app.py

import streamlit as st
from agents.agent_setup import setup_agents
import openai
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Set OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Configure logging
logging.basicConfig(
    filename='financial_agent.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)

# Initialize Agents
instruction_path = "instructions/"
multi_ai_agent = setup_agents(instruction_path)

# Streamlit App Layout
st.set_page_config(page_title="Financial Agent", layout="wide")

st.title("Financial Agent")

st.markdown("""
    The Financial Agent is a multi-modal AI-powered assistant designed to:
    - Search the web for financial news and analyst insights.
    - Retrieve and summarize financial data such as stock prices, fundamentals, and company news.
""")

# Sidebar for Navigation and Info
st.sidebar.header("About")
st.sidebar.info("""
    **Financial Agent** helps you:
    - Search the web for financial news and analyst insights.
    - Retrieve and summarize financial data like stock prices, fundamentals, and company news.
    
    **How to Use:**
    1. Enter your financial query in the input box.
    2. Click "Submit" to receive a summarized response.
    3. Optionally, download the response for future reference.
""")

# User Input
query = st.text_input("Enter your financial query:",
                      "Summarize the analyst response and share the latest news for NVIDIA.")

# Submit Button
if st.button("Submit"):
    with st.spinner("Processing..."):
        try:
            # Initialize an empty placeholder for the response
            response_placeholder = st.empty()
            response_text = ""

            # Generate Response
            response_generator = multi_ai_agent.print_response(
                query, stream=True)

            # Iterate over the streaming response and update the placeholder
            for chunk in response_generator:
                if chunk is not None:
                    response_text += chunk
                    response_placeholder.markdown(response_text)
                else:
                    st.warning("Received an empty response chunk.")

            if not response_text.strip():
                st.warning(
                    "The AI didn't return any information. Please try a different query.")
            else:
                # Log the response
                logging.info(f"Query: {query} | Response: {response_text}")

                # Save Response to File
                with open("response.txt", "w") as file:
                    file.write(response_text)

                # Provide Download Option
                st.download_button(
                    label="Download Response",
                    data=response_text,
                    file_name="response.txt",
                    mime="text/plain"
                )

        except Exception as e:
            st.error(f"An error occurred: {e}")
            logging.error(f"Error processing query '{query}': {e}")

# Debugging: Optionally display model IDs
if st.checkbox("Show Agent Model IDs"):
    try:
        web_search_model_id = multi_ai_agent.team[0].model.id
        finance_model_id = multi_ai_agent.team[1].model.id
        st.write("Web Search Agent model:", web_search_model_id)
        st.write("Financial Agent model:", finance_model_id)
    except AttributeError:
        st.write("Unable to retrieve model IDs.")
