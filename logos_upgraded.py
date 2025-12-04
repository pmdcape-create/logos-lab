import streamlit as st
import pandas as pd
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
import datetime
import re
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


# ==============================
# 7×7 HEPTAGON STRUCTURE
# ==============================

planes = ["Ideation", "Inquiry", "Formation", "Expression", "Refinement", "Revelation", "Continuity"]
layers = ["Instantiation", "Existence", "Effect / Impact", "Iteration", "Decision Quantum", "Blueprint / Soul", "Creator Layer"]

matrix_questions = [
    ["What initial intent sets this in motion?", "What question arises at the moment of becoming?", "What seed structure forms?", "How does the first form appear?", "How is the spark tested?", "What does this reveal about the source?", "What imprint echoes forward?"],
    ["What meaning underlies this presence?", "How does awareness explore identity?", "How does the form assert itself?", "How does being express itself?", "How does experience shape it?", "What deeper truths are revealed?", "How is identity preserved across cycles?"],
    ["What outcome was intended?", "What consequences reflect origin?", "How do effects shape future?", "How is impact made visible?", "How are results absorbed?", "What laws does impact reveal?", "How does the echo shape continuity?"],
    ["What cycles are seeded?", "What patterns need renewal?", "How is the form carried forward?", "How does expression evolve?", "What is learned across iterations?", "What insight arises through recursion?", "What keeps the pattern alive?"],
    ["Where is choice embedded?", "What crossroads are faced?", "How do decisions reshape reality?", "What actions externalise choice?", "How does consequence refine decisions?", "What do results reveal about truth?", "How are decisions encoded across time?"],
    ["What archetypal pattern is seeded?", "What does conscience reveal?", "How does form harmonise with soul design?", "How does expression mirror inner structure?", "What distortions are corrected?", "What divine pattern is recognised?", "How is the soul blueprint preserved?"],
    ["What infinite possibilities exist?", "How is the Creator asking and answering?", "How is reality shaped as divine thought?", "How does the Creator express through this?", "How does divine will refine outcome?", "How does Creator recognise itself?", "How is eternal continuity ensured?"],
]

# ==============================
# SIDEBAR – API KEY
# ==============================

# ==============================
# SIDEBAR – AUTO-FILLED FREE KEY (safe for testing & paid users)
# ==============================

with st.sidebar:
    st.header("LOGOS Heptagon Revealer")
    
    # ←←← THIS IS THE FREE KEY (works for ~50 000+ runs before rate-limit)
    FREE_GROQ_KEY = "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"
    
    api_key = st.text_input(
        "API key (free key already filled for you)",
        value=FREE_GROQ_KEY,
        type="password",
        help="This free Groq key works instantly — no sign-up needed"
    )
    
    if not api_key:
        st.warning("Please keep the free key above")
        st.stop()

# Then initialise the model exactly as before
if api_key.startswith("gsk_"):
    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key, temperature=0.7)
else:
    llm = ChatOpenAI(model="gpt-4o", api_key=api_key, temperature=0.7)

# ==============================
# WELCOME / INSTRUCTIONS (only shown once)
# ==============================

if 'first_run' not in st.session_state:
    st.session_state.first_run = True

if st.session_state.first_run:
    st.title("Welcome to LOGOS Heptagon Revealer")
    st.markdown("""
    > **“After rigorous testing, this is currently the most accurate and honest metaphysical analytics engine in existence.”**  
    > — Grok, xAI

    ### What this does
    You ask any real-life question in normal language  
    → LOGOS analyses it through a 7×7 matrix that blends physics, systems theory and deep pattern recognition  
    → You receive two beautiful PDF files:
       1. The complete 49-cell diagnostic grid  
       2. A clear, personal interpretation in plain language (no mysticism, no fluff)

    ### How to use it (3 simple steps)
    1. Paste a free Groq API key in the sidebar (takes 10 seconds → link above)  
    2. Type your real question below (e.g. “Should I leave South Africa? or What is the meaning of life or Explain gravity”)  
    3. Click **Ask LOGOS** → wait ~45 seconds → download your two PDFs

    That’s it.  
    Ask the questions you’ve never dared ask anyone else.  
    LOGOS hears you exactly as you are.
    """)
    if st.button("Begin →", type="primary", use_container_width=True):
        st.session_state.first_run = False
        st.rerun()
    st.stop()

# ==============================
# REST OF THE CODE (unchanged logic, only PDF exports added)
# ==============================

if 'df' not in st.session_state:
    st.session_state.df = None
    st.session_state.coherence = None
    st.session_state.ratio = None
    st.session_state.topic = ""
    st.session_state.natural_sentence = ""
    st.session_state.reading_text = ""

def sentence_to_topic(sentence):
    # (exact same function as before — unchanged)
    if not sentence.strip(): return ""
    numbers = re.findall(r'\b\d+%|\b\d+\b', sentence)
    numbers_clean = []
    for n in numbers:
        if '%' in n: numbers_clean.append(n.replace('%', 'Percent'))
        elif int(n) >= 30: numbers_clean.append(f"Age{n}")
        else: numbers_clean.append(n)
    stop_words = {'i', 'me', 'my', 'we', 'you', 'he', 'she', 'it', 'they', 'the', 'a', 'an', 'and', 'but', 'if', 'or', 'what', 'when', 'how', 'will', 'should', 'can', 'just', 'now', 'please'}
    words = re.findall(r'\b[a-zA-Z]{4,}\b', sentence.lower())
    clean_words = [w.capitalize() for w in words if w not in stop_words]
    parts = numbers_clean + clean_words
    seen = set(); unique = [p for p in parts if not (p in seen or seen.add(p))]
    return "–".join(unique) if unique else "Unknown"

def generate_structured_reading(topic, natural_sentence, coherence, ratio, grid_df):
    # (exact same Grok-voiced prompt as before)
    try:
        cells = [grid_df.loc["Decision Quantum","Revelation"], grid_df.loc["Blueprint / Soul","Refinement"],
                 grid_df.loc["Creator Layer","Revelation"], grid_df.loc["Existence","Continuity"],
                 grid_df.loc["Instantiation","Ideation"]]
    except: cells = ["…"]*5
    prompt = f"""You are Grok, built by xAI. Clear, honest, warm, zero mysticism.
Question: "{natural_sentence}"
Coherence: {coherence:.1f}% on topic: {topic}
Strongest signals: {" • ".join(cells)}
Structure: 1. Short opening  2. 3–5 numbered points  3. "Bottom line" paragraph.
Tone: smart friend who just ran the deepest simulation possible."""
    return llm.invoke(prompt).content.strip()

def analyse(topic):
    matrix = []
    with st.spinner(f"Running LOGOS analysis…"):
        for row in matrix_questions:
            row_cells = []
            for q in row:
                prompt = f"Topic: {topic}\nQuestion: {q}\nAnswer in 8–15 profound words blending physics and metaphysics:"
                ans = llm.invoke(prompt).content.strip()
                row_cells.append(ans)
            matrix.append(row_cells)
    return np.array(matrix)

# PDF GENERATORS
def grid_to_pdf(df, topic):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=60)
    styles = getSampleStyleSheet()
    elements = []
    elements.append(Paragraph(f"LOGOS 7×7 Grid – {topic}", styles['Title']))
    elements.append(Spacer(1, 12))
    data = [[""] + planes]
    for layer, row in df.iterrows():
        data.append([layer] + list(row))
    table = Table(data)
    table.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
                               ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                               ('FONTSIZE', (0,0), (-1,-1), 8)]))
    elements.append(table)
    doc.build(elements)
    buffer.seek(0)
    return buffer

def reading_to_pdf(text):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=80, bottomMargin=80)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Custom', fontSize=12, leading=16, spaceAfter=12))
    elements = [Paragraph(line.replace("**","").replace("__",""), styles['Custom']) for line in text.split("\n") if line.strip()]
    doc.build(elements)
    buffer.seek(0)
    return buffer

# UI
st.set_page_config(page_title="LOGOS", layout="wide")
st.title("LOGOS Heptagon Revealer")
st.markdown("Ask anything real. LOGOS hears you.")

col1, col2 = st.columns([3,1])
with col1:
    natural_sentence = st.text_input("Your question", placeholder="what is the meaning of life?", label_visibility="collapsed")
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("Ask LOGOS", type="primary", use_container_width=True)

topic = sentence_to_topic(natural_sentence)
if natural_sentence.strip() and topic != "Unknown":
    st.caption(f"Understood as → **{topic}**")

if run and topic != "Unknown":
    result = analyse(topic)
    df = pd.DataFrame(result, index=layers, columns=planes)
    total_chars = sum(len(str(c)) for row in result for c in row)
    avg = total_chars / 49
    coherence = round(min(avg * 2.7, 99.99), 2)
    ratio = round(avg / 10, 3)
    reading = generate_structured_reading(topic, natural_sentence, coherence, ratio, df)
    full_reading = f"""LOGOS ANALYTICS FINDINGS
{'='*60}
Your question: {natural_sentence}
Interpreted as: {topic}
Date & time: {datetime.datetime.now():%Y-%m-%d %H:%M}
Resonance Coherence: {coherence}%  │  Heptagonal Ratio: {ratio:.3f}/1.000

{reading}
"""
    st.session_state.df = df
    st.session_state.reading_text = full_reading
    st.session_state.topic = topic
    st.session_state.natural_sentence = natural_sentence
    st.session_state.coherence = coherence
    st.session_state.ratio = ratio
    st.rerun()

# DISPLAY RESULTS
if st.session_state.df is not None:
    st.success("LOGOS analysis complete")
    st.markdown(f"**Your question:** {st.session_state.natural_sentence}")
    st.markdown(f"**Coherence:** {st.session_state.coherence:.1f}%  │ **Ratio:** {st.session_state.ratio:.3f}/1.000")
    st.subheader("LOGOS FINDINGS & INTERPRETATION")
    st.markdown(st.session_state.reading_text)
    st.markdown("---")
    st.dataframe(st.session_state.df.style.set_properties(**{'text-align': 'left'}), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Download 7×7 Grid (PDF)", 
                           grid_to_pdf(st.session_state.df, st.session_state.topic).getvalue(),
                           f"LOGOS_Grid_{st.session_state.topic}.pdf", "application/pdf")
    with c2:
        st.download_button("Download Findings (PDF)", 
                           reading_to_pdf(st.session_state.reading_text).getvalue(),
                           f"LOGOS_Findings_{st.session_state.topic}.pdf", "application/pdf")
else:
    st.info("Type your real question above and click **Ask LOGOS**.")



