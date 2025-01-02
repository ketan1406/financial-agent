# app.py

import streamlit as st
from agents.agent_setup import setup_agents
import openai
import os
from dotenv import load_dotenv
import logging
import json
import pandas as pd
from jsonschema import validate, ValidationError
import time

# Constants
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# Load environment variables
load_dotenv()

# Set OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Configure logging
logging.basicConfig(
    filename='financial_agent.log',
    level=logging.DEBUG,  # Set to DEBUG for detailed logs
    format='%(asctime)s %(levelname)s:%(message)s'
)

# Initialize Agents
instruction_path = "instructions/"
multi_ai_agent = setup_agents(instruction_path)


def extract_json_from_response(response):
    """
    Extracts the JSON string from the RunResponse object.
    Modify this function based on the actual structure of RunResponse.
    """
    try:
        # Inspect the response object
        logging.debug(f"Response Type: {type(response)}")
        logging.debug(f"Response Attributes: {dir(response)}")

        # Attempt to extract content
        if hasattr(response, 'content') and isinstance(response.content, (str, bytes)):
            json_str = response.content
            if isinstance(json_str, bytes):
                json_str = json_str.decode('utf-8')
        elif hasattr(response, 'text') and isinstance(response.text, str):
            json_str = response.text
        elif hasattr(response, 'json') and callable(getattr(response, 'json')):
            json_obj = response.json()
            if isinstance(json_obj, dict):
                json_str = json.dumps(json_obj)
            else:
                json_str = str(json_obj)
        else:
            # Fallback to string representation
            json_str = str(response)

        # Log first 100 chars
        logging.debug(f"Extracted JSON String: {json_str[:100]}...")
        return json_str
    except Exception as e:
        logging.error(f"Error extracting JSON from response: {e}")
        return None


def main():
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
                          "Summarize the analyst response and share the latest news for Tesla (NASDAQ: TSLA).")

    # Submit Button
    if st.button("Submit"):
        with st.spinner("Processing..."):
            response_json = None
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    # Use the agent's run method to get the response directly
                    response = multi_ai_agent.run(query)

                    # Extract JSON string from RunResponse
                    json_str = extract_json_from_response(response)

                    if not json_str:
                        raise ValueError("No content extracted from response.")

                    # Attempt to parse the response as JSON
                    try:
                        response_json = json.loads(json_str)
                        logging.debug(f"Parsed JSON: {response_json}")
                    except json.JSONDecodeError as jde:
                        logging.warning(
                            f"Attempt {attempt}: JSON Decode Error - {jde.msg}")
                        logging.debug(f"Full Response Content: {json_str}")
                        response_json = None

                    # Validate JSON Schema
                    if response_json:
                        schema = {
                            "type": "object",
                            "properties": {
                                "analyst_recommendations": {
                                    "type": "object",
                                    "properties": {
                                        "Strong Buy": {"type": "integer"},
                                        "Buy": {"type": "integer"},
                                        "Hold": {"type": "integer"},
                                        "Sell": {"type": "integer"},
                                        "Strong Sell": {"type": "integer"}
                                    },
                                    "required": ["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"]
                                },
                                "latest_news": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "title": {"type": "string"},
                                            "summary": {"type": "string"},
                                            "source": {"type": "string"},
                                            "url": {"type": "string", "format": "uri"}
                                        },
                                        "required": ["title", "summary", "source", "url"]
                                    }
                                }
                            },
                            "required": ["analyst_recommendations", "latest_news"]
                        }
                        try:
                            validate(instance=response_json, schema=schema)
                            break  # Successful parsing and validation
                        except ValidationError as ve:
                            logging.warning(
                                f"Attempt {attempt}: Schema Validation Error - {ve.message}")
                            response_json = None
                            break  # Don't retry if schema is incorrect
                except Exception as e:
                    logging.error(
                        f"Attempt {attempt}: Error processing query '{query}': {e}")
                    response_json = None
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_DELAY)

            if response_json:
                # Log the JSON response
                logging.info(f"Query: {query} | Response: {response_json}")

                st.markdown("### Analyst Recommendations")
                # Display Analyst Recommendations
                analyst_recs = response_json.get("analyst_recommendations", {})
                if analyst_recs:

                    # Optional: Visualize as Bar Chart
                    df_recs = pd.DataFrame(list(analyst_recs.items()), columns=[
                                           'Recommendation', 'Count'])

                    # Display as a Streamlit table
                    st.table(df_recs)

                    # Optional: Visualize as Bar Chart
                    st.bar_chart(data=df_recs.set_index('Recommendation'))
                else:
                    st.warning(
                        "No analyst recommendations found in the response.")

                # Display Latest News
                st.markdown("### Latest Tesla News")
                latest_news = response_json.get("latest_news", [])
                if latest_news:
                    for news in latest_news:
                        title = news.get("title", "No Title")
                        summary = news.get("summary", "No Summary")
                        source = news.get("source", "Unknown")
                        url = news.get("url", "#")
                        with st.expander(title):
                            st.markdown(f"{summary}")
                            st.markdown(
                                f"[Read more]({url}) (Source: {source})")
                else:
                    st.warning("No latest news found in the response.")

                # Save Response to File
                with open("response.json", "w") as file:
                    json.dump(response_json, file, indent=4)

                # Provide Download Option
                st.download_button(
                    label="Download Response as Text",
                    data=json.dumps(response_json, indent=4),
                    file_name="response.txt",
                    mime="text/plain"
                )
            else:
                st.warning(
                    "The AI didn't return a valid JSON response. Please try a different query.")


if __name__ == "__main__":
    main()
