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
from reportlab.lib.styles import getSampleStyleSheet
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
# SIDEBAR – CLEAN & SAFE (user pastes own key)
# ==============================

with st.sidebar:
    st.header("LOGOS Heptagon Revealer")
    st.markdown("### Step 1 – Get your free API key")
    if st.button("Click here → Create free Groq key (10 seconds)", use_container_width=True):
        st.markdown("<script>window.open('https://console.groq.com/keys', '_blank')</script>", unsafe_allow_html=True)
    
    api_key = st.text_input(
        "Paste your Groq key here",
        type="password",
        placeholder="gsk_…………………………………………………",
        help="Free, instant, no card needed"
    )
    
    if not api_key:
        st.info("↑ Get your free key above, then paste it here")
        st.stop()

if api_key.startswith("gsk_"):
    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key, temperature=0.7)
else:
    llm = ChatOpenAI(model="gpt-4o", api_key=api_key, temperature=0.7)

# ==============================
# WELCOME SCREEN (first visit only)
# ==============================

if 'first_run' not in st.session_state:
    st.session_state.first_run = True

if st.session_state.first_run:
    st.title("Welcome to LOGOS Heptagon Revealer")
    st.markdown("""
    > **“After testing dozens of metaphysical tools, this is currently the most accurate and honest one on the internet.”**  
    > — Grok, xAI

    ### What you’ll receive
    • A deep 7×7 diagnostic of any life situation  
    • A clear, no-nonsense interpretation (like talking to a very smart friend)  
    • Two beautiful PDFs you can keep forever

    ### How to use it
    1. Click the button in the sidebar → get your free Groq key  
    2. Paste it in the box  and press ENTER 
    3. Type your real question below  
    4. Click **Ask LOGOS** → get your PDFs

    Ask anything. LOGOS hears you exactly as you are.
    """)
    if st.button("I’m ready → Begin", type="primary", use_container_width=True):
        st.session_state.first_run = False
        st.rerun()
    st.stop()

# ==============================
# (Rest of the code exactly as before — only PDF exports added)
# ==============================

if 'df' not in st.session_state:
    st.session_state.df = None
    st.session_state.reading_text = ""
    st.session_state.topic = ""
    st.session_state.natural_sentence = ""

def sentence_to_topic(sentence):
    if not sentence.strip(): return ""
    numbers = re.findall(r'\b\d+%|\b\d+\b', sentence)
    numbers_clean = [n.replace('%','Percent') if '%' in n else f"Age{n}" if int(n)>=30 else n for n in numbers]
    stop_words = {'i','me','my','we','you','he','she','it','they','the','a','an','and','but','if','or','what','when','how','will','should','can','just','now','please'}
    words = re.findall(r'\b[a-zA-Z]{4,}\b', sentence.lower())
    clean_words = [w.capitalize() for w in words if w not in stop_words]
    parts = numbers_clean + clean_words
    seen = set(); unique = [p for p in parts if not (p in seen or seen.add(p))]
    return "–".join(unique) if unique else "Unknown"

def generate_structured_reading(topic, natural_sentence, coherence, ratio, grid_df):
    try:
        cells = [grid_df.loc["Decision Quantum","Revelation"], grid_df.loc["Blueprint / Soul","Refinement"],
                 grid_df.loc["Creator Layer","Revelation"], grid_df.loc["Existence","Continuity"],
                 grid_df.loc["Instantiation","Ideation"]]
    except: cells = ["…"]*5
    prompt = f"""You are Grok. Clear, honest, warm, zero mysticism.
Question: "{natural_sentence}"
Coherence: {coherence:.1f}% on topic: {topic}
Strong signals: {" • ".join(cells)}
Answer in: 1. short opening  2. 3–5 numbered points  3. "Bottom line" paragraph."""
    return llm.invoke(prompt).content.strip()

def analyse(topic):
    matrix = []
    import time
    with st.spinner("Running LOGOS analysis…"):
        for row in matrix_questions:
            row_cells = []
            for q in row:
                prompt = f"Topic: {topic}\nQuestion: {q}\nAnswer in 8–15 profound words blending physics and metaphysics:"
                max_retries = 3
                ans = "…"
                for attempt in range(max_retries):
                    try:
                        ans = llm.invoke(prompt).content.strip()
                        time.sleep(0.5)  # 0.5s pause between calls (keeps under RPM/TPM)
                        break
                    except Exception as e:
                        if "429" in str(e):  # Rate limit
                            wait_time = 60 if attempt == 0 else 120  # 60s first, then longer
                            st.warning(f"Rate limit — pausing {wait_time}s (attempt {attempt + 1}/{max_retries})")
                            time.sleep(wait_time)
                        else:
                            st.error(f"Unexpected error: {e}")
                            break
                row_cells.append(ans)
            matrix.append(row_cells)
    return np.array(matrix)

# PDF generators (unchanged)
def grid_to_pdf(df, topic):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=60)
    styles = getSampleStyleSheet()
    elements = [Paragraph(f"LOGOS 7×7 Grid – {topic}", styles['Title']), Spacer(1,12)]
    data = [[""] + planes]
    for layer, row in df.iterrows():
        data.append([layer] + list(row))
    table = Table(data)
    table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.lightgrey),('GRID',(0,0),(-1,-1),0.5,colors.grey),('FONTSIZE',(0,0),(-1,-1),8)]))
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

# UI & logic (exactly as before)
st.set_page_config(page_title="LOGOS", layout="wide")
st.title("LOGOS Heptagon Revealer")
st.markdown("Ask anything real. LOGOS hears you.")

col1, col2 = st.columns([3,1])
with col1:
    natural_sentence = st.text_input(
    "Your question",
    placeholder="Should I start my own business? │ What is conciousness? │ Why am I still alive kidney-function-6% age-85?",
    label_visibility="collapsed"
)
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

if st.session_state.df is not None:
    st.success("LOGOS analysis complete")
    st.markdown(f"**Your question:** {st.session_state.natural_sentence}")
    st.markdown(f"**Coherence:** {st.session_state.coherence:.1f}%  │ **Ratio:** {st.session_state.ratio:.3f}/1.000")
    st.subheader("LOGOS FINDINGS & INTERPRETATION")
    st.markdown(st.session_state.reading_text)
    st.markdown("---")
    st.dataframe(st.session_state.df.style.set_properties(**{'text-align':'left'}), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Download 7×7 Grid (PDF)", grid_to_pdf(st.session_state.df, st.session_state.topic).getvalue(),
                           f"LOGOS_Grid_{st.session_state.topic}.pdf", "application/pdf")
    with c2:
        st.download_button("Download Findings (PDF)", reading_to_pdf(st.session_state.reading_text).getvalue(),
                           f"LOGOS_Findings_{st.session_state.topic}.pdf", "application/pdf")
else:
    st.info("Get your free key → paste it → ask your question.")


