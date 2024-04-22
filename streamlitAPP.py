import os
import json as js
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.multigen.utils import read_file,get_table_data
from langchain_openai import ChatOpenAI
import streamlit as st
from langchain_community.callbacks import get_openai_callback
from src.multigen.multigen import generate_evaluate_chain
from src.multigen.logger import logging

with open('/Users/sangeethreddy/multigen/Response.json','r') as file:
    RESPONSE_JSON = js.load(file)

st.title("MCQs Creator Application with Langchain")

with st.form("user_inputs"):
    upload_file = st.file_uploader("Upload a PDF or txt file")
    mcq_count = st.number_input("No. of MCQs", min_value=3, max_value=50)
    subject = st.text_input("Insert Subject", max_chars=20)
    # tone = st.text_input("Complexity Level of Questions",
    #                      max_chars=20, placeholder="Simple")
    st.caption(
        "Please enter the complexity level of questions (e.g., Simple, Advanced):")
    tone = st.text_input("Complexity Level of Questions", max_chars=20)

    button = st.form_submit_button("Create MCQs")

    if button and upload_file is not None and mcq_count and subject and tone:
        with st.spinner("loading..."):
            try:
                text = read_file(upload_file)
                with get_openai_callback() as cb:
                    response = generate_evaluate_chain(
                        {
                            "text": text,
                            "number": mcq_count,
                            "subject": subject,
                            "tone": tone,
                            "response_json": js.dumps(RESPONSE_JSON)
                        }
                    )

            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)
                st.error("Error")

            else:
                print(f"Total Tokens:{cb.total_tokens}")
                print(f"Prompt Tokens:{cb.prompt_tokens}")
                print(f"Completion Tokens:{cb.completion_tokens}")
                print(f"Total Cost:{cb.total_cost}")
                if isinstance(response, dict):
                    quiz = response.get("quiz", None)
                    if quiz is not None:
                        table_data = get_table_data(quiz)
                        if table_data is not None:
                            df = pd.DataFrame(table_data)
                            df.index = df.index+1
                            st.table(df)
                            if "Review" in response:
                                st.text_area(label="Review", value=response["Review"])
                            else:
                                st.warning("Review not available")
                        else:
                            st.error("Error in the table data")
                    else:
                        st.write(response)
                else:
                    st.error("Invalid response format")
