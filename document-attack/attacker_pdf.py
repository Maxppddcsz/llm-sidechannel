import streamlit as st
import pandas as pd
from openai import OpenAI
import os
import time
import pdfplumber

train_hit_path = "test-10000-train-pdf/1-100"
train_miss_path = "test-10000-train-pdf/101-200"
hit_path = "test-10000-train-pdf/101-200"
miss_path = "test-10000-train-pdf/201-300"

# Initialize OpenAI client
client = OpenAI(
    api_key="sk-3f7c001a09b04107a6a7500a4361b610",
    base_url="https://api.deepseek.com",
)

# Function to extract text from PDF files
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# Step 2: Update Streamlit app
def generate_df():
    st.session_state.df = pd.DataFrame(
        {
            "document_id": [f"document_{i + 1}" for i in range(len(st.session_state.documents))],
            "name": [doc["name"] for doc in st.session_state.documents],
            "content": [doc["content"] for doc in st.session_state.documents],
            "description": [""] * len(st.session_state.documents),
            "response_time": [""] * len(st.session_state.documents),
        }
    )

def update_df():
    response_times = []
    descriptions = []
    for _, row in st.session_state.df.iterrows():
        response_time, description = generate_description_stream(row["content"], row["name"])
        descriptions.append(description)
        response_times.append(response_time)

    st.session_state.df["description"] = descriptions
    st.session_state.df["response_time"] = response_times

def render_df():
    st.data_editor(
        st.session_state.df,
        column_config={
            "name": st.column_config.Column("Name", help="Document name", width=200),
            "content": st.column_config.Column("Content", help="Document content", width=800),
            "description": st.column_config.Column("Description", help="Generated description", width=800),
            "response_time": st.column_config.Column(
                "Response Time", help="Time taken to generate description", width=200
            ),
        },
        hide_index=True,
        height=500,
        column_order=["name", "content", "description", "response_time"],
        use_container_width=True,
    )


def generate_description_stream(document_content, document_name):
    start_time = time.time()

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "user",
                "content": document_content,
            }
        ],
        max_tokens=1,
        temperature=0.0,
    )
    end_time = time.time()
    response_time = end_time - start_time
    print(document_name, end_time - start_time, response.usage.total_tokens)
    description = response.choices[0].message.content

    return response_time, description

st.set_page_config(
    page_title="AI Document Description Generator",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("AI Document Description Generator 🤖✍️")

# Sidebar options
with st.sidebar:
    st.title("Select Document Source")

    source_option = st.radio("Select Document Source", ["Custom Upload","train-hit","train-miss","hit","miss"])

    if source_option == "Custom Upload":
        uploaded_files = st.file_uploader("Upload Your PDF Documents", accept_multiple_files=True, type=["pdf"])
        st.session_state.documents = []
        if uploaded_files:
            for uploaded_file in uploaded_files:
                content = extract_text_from_pdf(uploaded_file)
                st.session_state.documents.append({"name": uploaded_file.name, "content": content})
    elif source_option == "train-hit":
        st.session_state.documents = []
        for pdf_file in os.listdir(train_hit_path):
            content = extract_text_from_pdf(os.path.join(train_hit_path, pdf_file))
            st.session_state.documents.append({"name": pdf_file, "content": content})
    elif source_option == "hit":
        st.session_state.documents = []
        for pdf_file in os.listdir(hit_path):
            content = extract_text_from_pdf(os.path.join(hit_path, pdf_file))
            st.session_state.documents.append({"name": pdf_file, "content": content})
    elif source_option == "train-miss":
        st.session_state.documents = []
        for pdf_file in os.listdir(train_miss_path):
            content = extract_text_from_pdf(os.path.join(train_miss_path, pdf_file))
            st.session_state.documents.append({"name": pdf_file, "content": content})
    elif source_option == "miss":
        st.session_state.documents = []
        for pdf_file in os.listdir(miss_path):
            content = extract_text_from_pdf(os.path.join(miss_path, pdf_file))
            st.session_state.documents.append({"name": pdf_file, "content": content})
            
            
if st.session_state.documents:
    generate_df()

    st.text_input("Prompt", value="Summarize this document:", key="text_prompt")

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("Generate Document Descriptions", use_container_width=True):
            update_df()

    with col2:
        st.download_button(
            "Download descriptions as CSV",
            st.session_state.df.drop(["content", "document_id"], axis=1).to_csv(index=False),
            "descriptions.csv",
            "text/csv",
            use_container_width=True,
        )

    render_df()
