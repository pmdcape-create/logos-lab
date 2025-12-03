import streamlit as st
import pandas as pd
import numpy as np
from sympy import Matrix, latex, symbols
import re
from collections import Counter
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import BaseRetriever  # Placeholder for RAG
import os
import io  # For in-memory Excel

# Enable LangSmith for observability (optional)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = st.secrets.get("LANGCHAIN_API_KEY", "dummy")

# Data (unchanged from before)
laws = [...]  # (Omit for brevity; copy from previous code)
planes = ["Ideation", "Inquiry", "Formation", "Expression", "Refinement", "Revelation", "Continuity"]
layers = ["Instantiation", "Existence", "Effect / Impact", "Iteration", "Decision Quantum", "Blueprint / Soul", "Creator Layer"]
matrix_questions = [...]  # (Omit; copy full list)

# Custom Theme (2025 feature)
st.set_page_config(
    page_title="Upgraded LOGOS Lab",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="expanded"
)
if "dark" in st.get_option("theme"):  # Dynamic theming
    st.markdown("""
    <style>
        .stApp { background-color: #0e1117; }
        .stMetric { color: #ffffff; }
    </style>
    """, unsafe_allow_html=True)

# LLM Setup
@st.cache_resource
def get_llm():
    api_key = st.sidebar.text_input("OpenAI/Groq API Key:", type="password")
    if "openai" in api_key.lower() or not api_key:
        return ChatOpenAI(model="gpt-4o", api_key=api_key, temperature=0.7)
    else:
        return ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key, temperature=0.7)

llm = get_llm()

# RAG Placeholder (upload workbook as context)
uploaded_file = st.sidebar.file_uploader("Upload Workbook for RAG Context (optional)")
context_docs = []  # Parse uploaded Excel/CSV into strings
if uploaded_file:
    df_context = pd.read_excel(uploaded_file)
    context_docs = df_context.astype(str).values.flatten().tolist()

# Auto-Fill Chain
fill_prompt = PromptTemplate(
    input_variables=["question", "topic", "context"],
    template="""You are a LOGOS analyst. For the topic '{topic}', answer this guiding question concisely, drawing from context if relevant: '{question}'.
    Ensure alignment with spiritual/physical duality. Context: {context}"""
)
fill_chain = LLMChain(llm=llm, prompt=fill_prompt)

@st.cache_data
def auto_fill_matrix(topic, context):
    matrix_data = []
    with st.spinner("AI filling matrix..."):
        for i, row in enumerate(matrix_questions):
            row_filled = []
            for q in row:
                response = fill_chain.run(question=q, topic=topic, context=" ".join(context))
                row_filled.append(response)
            matrix_data.append(row_filled)
    return np.array(matrix_data)

# Enhanced Laws Check with LLM
def check_laws_with_llm(matrix_str):
    violations = []
    law_prompt = PromptTemplate(
        input_variables=["matrix", "law_desc"],
        template="Check if this matrix violates: '{law_desc}'. Matrix: {matrix}. Respond 'Violation: [details]' or 'Pass'."
    )
    law_chain = LLMChain(llm=llm, prompt=law_prompt)
    for law in laws:
        result = law_chain.run(matrix=" ".join(matrix_str.flatten()), law_desc=law["Description"])
        if "violation" in result.lower():
            violations.append(f"Law {law['ID']}: {result}")
    return violations

# Advanced Vectorization with TF-IDF
def advanced_vectorize(matrix_data):
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([" ".join(row) for row in matrix_data])
    layer_vectors = tfidf_matrix.mean(axis=1).toarray()  # Average per layer

    # Enhanced M: Cosine similarity + eigenvalue for resonance
    M_advanced = np.zeros((7,7))
    for i in range(7):
        for j in range(7):
            if np.any(layer_vectors[i]) and np.any(layer_vectors[j]):
                sim = 1 - cosine(layer_vectors[i], layer_vectors[j])
                M_advanced[i,j] = sim
    M_advanced /= np.max(M_advanced) if np.max(M_advanced) > 0 else 1

    # Resonance: Eigenvalue of M (emergent complexity)
    eigenvalues = np.linalg.eigvals(M_advanced)
    resonance = np.max(np.real(eigenvalues))

    return M_advanced, layer_vectors, resonance

# Plot for Math Demo
def plot_resonance(M, resonance):
    fig, ax = plt.subplots()
    im = ax.imshow(M, cmap='viridis')
    ax.set_title(f'Resonance Eigenvalue: {resonance:.2f}')
    plt.colorbar(im)
    return fig

st.title("🔷 Upgraded LOGOS Model Logic Laboratory")

# Sidebar
with st.sidebar:
    st.header("AI Settings")
    st.info("Paste API key for auto-fill magic.")
    topic = st.text_input("Topic for Analysis:")

if st.button("Run Upgraded LOGOS Analysis"):
    if topic:
        matrix_data = auto_fill_matrix(topic, context_docs)
        df = pd.DataFrame(matrix_data, index=layers, columns=planes)

        # Laws Check
        violations = check_laws_with_llm(matrix_data)

        # Display
        st.subheader("AI-Populated 7x7 LOGOS Matrix")
        st.dataframe(df, use_container_width=True)

        st.subheader("Systemic Laws Coherence (AI-Checked)")
        if violations:
            for v in violations:
                st.warning(v)
        else:
            st.success("Coherent across all laws!")

        # Math Demo
        st.subheader("Advanced Math & Vectorization")
        M, layer_vecs, resonance = advanced_vectorize(matrix_data)
        st.write("TF-IDF Transition Matrix M:")
        st.dataframe(pd.DataFrame(M, index=layers, columns=planes))

        fig = plot_resonance(M, resonance)
        st.pyplot(fig)

        # SymPy (unchanged)
        sym_M = Matrix(np.round(M, 2))
        s = np.linalg.norm(layer_vecs, axis=1)
        s_norm = s / np.max(s) if np.max(s) > 0 else s
        sym_s = Matrix(np.round(s_norm, 2))
        sym_result = sym_M * sym_s
        st.latex(f"\\vec{{result}} = M \\cdot \\vec{{s}} = {latex(sym_result)}")

        T, r = symbols('T r')
        st.latex(f"\\nabla T = -\\frac{{\\partial T}}{{\\partial r}} \\quad (Gravity Tension)")

        # Exports
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Matrix')
            pd.DataFrame(M).to_excel(writer, sheet_name='Vectors')
        st.download_button("Download Enhanced Excel", output.getvalue(), "logos_upgraded.xlsx")

    else:
        st.error("Enter a topic!")

st.markdown("---")
st.caption("Upgraded Dec 2025: LLM Auto-Fill, RAG, Observability. Built with LangChain & 2025 Streamlit features.")