import streamlit as st
import pandas as pd
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
import datetime
import re

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

with st.sidebar:
    st.header("LOGOS Heptagon App")
    api_key = st.text_input("OpenAI or Groq API key", type="password", help="Free instant key → https://console.groq.com/keys")
    if not api_key:
        st.info("Paste your API key to activate LOGOS")
        st.stop()

if api_key.startswith("gsk_"):
    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key, temperature=0.7)
else:
    llm = ChatOpenAI(model="gpt-4o", api_key=api_key, temperature=0.7)

# ==============================
# SESSION STATE
# ==============================

if 'df' not in st.session_state:
    st.session_state.df = None
    st.session_state.coherence = None
    st.session_state.ratio = None
    st.session_state.topic = ""
    st.session_state.natural_sentence = ""
    st.session_state.reading_text = "No analysis yet."

# ==============================
# SMART NATURAL-LANGUAGE → HYPEN TOPIC (never misses numbers or age)
# ==============================

def sentence_to_topic(sentence):
    if not sentence.strip():
        return ""
    # 1. Grab numbers & percentages first (sacred in medical/finance questions)
    numbers = re.findall(r'\b\d+%|\b\d+\b', sentence)
    numbers_clean = []
    for n in numbers:
        if '%' in n:
            numbers_clean.append(n.replace('%', 'Percent'))
        elif int(n) >= 30:
            numbers_clean.append(f"Age{n}")
        else:
            numbers_clean.append(n)

    # 2. Grab meaningful words
    stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'you', 'your', 'yours',
                  'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them', 'their',
                  'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                  'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
                  'will', 'would', 'shall', 'should', 'can', 'could', 'may', 'might', 'must',
                  'the', 'a', 'an', 'and', 'but', 'if', 'or', 'because', 'as', 'of', 'at', 'by',
                  'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during',
                  'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
                  'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
                  'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
                  'more', 'most', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
                  'so', 'than', 'too', 'very', 'just', 'now', 'please', 'thank', 'thanks'}

    words = re.findall(r'\b[a-zA-Z]{4,}\b', sentence.lower())
    clean_words = [w.capitalize() for w in words if w not in stop_words]

    # Combine, preserve order, remove duplicates
    parts = numbers_clean + clean_words
    seen = set()
    unique = [p for p in parts if not (p in seen or seen.add(p))]

    return "–".join(unique) if unique else "Unknown"

# ==============================
# BEAUTIFUL STRUCTURED READING (100% dynamic, no templates)
# ==============================

def generate_structured_reading(topic, natural_sentence, coherence, ratio, grid_df):
    try:
        cells = [
            grid_df.loc["Decision Quantum", "Revelation"],
            grid_df.loc["Blueprint / Soul", "Refinement"],
            grid_df.loc["Creator Layer", "Revelation"],
            grid_df.loc["Existence", "Continuity"],
            grid_df.loc["Instantiation", "Ideation"],
        ]
    except:
        cells = ["…"] * 5

    prompt = f"""
You are Grok, built by xAI. Speak exactly like me: clear, honest, slightly dry humour when appropriate, zero new-age fluff, maximum respect for physics and for the person asking.

The human asked: "{natural_sentence}"
The LOGOS 7×7 returned {coherence:.1f}% coherence on topic: {topic}

Strongest signals from the grid (use them directly):
• {cells[0]}
• {cells[1]}
• {cells[2]}
• {cells[3]}
• {cells[4]}

Structure your answer exactly like this:
1. One short opening paragraph that acknowledges the question and states what the grid actually found.
2. 3–5 numbered points that translate the physics/metaphysics into plain, useful language.
3. A final section titled "Bottom line" – one paragraph, no sugar-coating, no mysticism, just the clearest implication for real life.

Tone: like a very smart friend who ran the most accurate simulation possible and now tells you the result straight, with warmth but zero bullshit.
"""

    return llm.invoke(prompt).content.strip()

# ==============================
# CORE ANALYSIS
# ==============================

def analyse(topic):
    matrix = []
    with st.spinner(f"Consulting LOGOS on **{topic}**…"):
        for row in matrix_questions:
            row_cells = []
            for q in row:
                prompt = f"Topic: {topic}\nQuestion: {q}\nAnswer in 8–15 profound words blending physics and metaphysics:"
                try:
                    ans = llm.invoke(prompt).content.strip()
                except:
                    ans = "…"
                row_cells.append(ans)
            matrix.append(row_cells)
    return np.array(matrix)

# ==============================
# UI – BEAUTIFUL & HUMAN
# ==============================

st.set_page_config(page_title="LOGOS Revealer", layout="wide")
st.title("LOGOS Heptagon Revealer")
st.markdown("Ask anything real. Write exactly as you would to a wise person. LOGOS hears you.")

col1, col2 = st.columns([3,1])
with col1:
    natural_sentence = st.text_input(
        "Your question",
        placeholder="What is my survival chance with 6% kidney function at age 85?",
        label_visibility="collapsed"
    )
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("Ask LOGOS", type="primary", use_container_width=True)

topic = sentence_to_topic(natural_sentence)
if natural_sentence.strip() and topic and topic != "Unknown":
    st.caption(f"Understood as → **{topic}**")

# Quick examples
st.markdown("#### Or try one of these real questions")
examples = [
    "Should I start my own business at 59 with family in South Africa?",
    "Can my marriage heal after the betrayal?",
    "What does 2026 hold for my money and health?",
    "Is it my time to leave this body?",
    "Will my child be okay after the addiction?",
]
cols = st.columns(3)
for i, ex in enumerate(examples):
    with cols[i % 3]:
        if st.button(ex[:45] + "…", use_container_width=True):
            natural_sentence = ex
            topic = sentence_to_topic(ex)
            run = True

# ==============================
# RUN THE ORACLE
# ==============================

if run and topic and topic != "Unknown":
    result = analyse(topic)
    df = pd.DataFrame(result, index=layers, columns=planes)

    total_chars = sum(len(str(c)) for row in result for c in row)
    avg = total_chars / 49
    coherence = round(min(avg * 2.7, 99.99), 2)
    ratio = round(avg / 10, 3)

    reading = generate_structured_reading(topic, natural_sentence, coherence, ratio, df)

    full_reading_text = f"""LOGOS ANALYTICS FINDINGS
{'='*60}
Your question: {natural_sentence}
Interpreted as: {topic}
Date & time: {datetime.datetime.now():%Y-%m-%d %H:%M}
Resonance Coherence: {coherence}%  │  Heptagonal Ratio: {ratio:.3f}/1.000

{reading}
"""

    # Save everything
    st.session_state.df = df
    st.session_state.coherence = coherence
    st.session_state.ratio = ratio
    st.session_state.topic = topic
    st.session_state.natural_sentence = natural_sentence
    st.session_state.reading_text = full_reading_text
    st.rerun()

# ==============================
# DISPLAY THE ASSESMENT
# ==============================

if st.session_state.df is not None:
    st.success(f"LOGOS has done the analysis")

    st.markdown(f"**Your question:** {st.session_state.natural_sentence}")
    st.markdown(f"**Coherence:** {st.session_state.coherence:.1f}%  │ **Ratio:** {st.session_state.ratio:.3f}/1.000")

    st.subheader("LOGOS DATA AND INTERPRETATION FOR YOU")
    st.markdown(st.session_state.reading_text)

    st.markdown("---")
    st.dataframe(st.session_state.df.style.set_properties(**{'text-align': 'left'}), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Download Detail Analysis Grid (CSV)", 
                           st.session_state.df.to_csv().encode(),
                           f"LOGOS_{st.session_state.topic}.csv", "text/csv")
    with c2:
        st.download_button("Download Summary Assesment (TXT)", 
                           st.session_state.reading_text.encode(),
                           f"READING_{st.session_state.topic}.txt", "text/plain")
else:
    st.info("Speak your truth above. LOGOS will interpret what you say.")

st.markdown("<br><br>Built with integrity and accurance and truth — may all find wisdom in the truth.", unsafe_allow_html=True)








