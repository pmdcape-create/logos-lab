import streamlit as st
import pandas as pd
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
import datetime
import re

# ==============================
# 7×7 HEPTAGON DATA
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
    st.header("LOGOS Heptagon Oracle")
    api_key = st.text_input("OpenAI or Groq API key", type="password", help="Free instant key → https://console.groq.com/keys")
    if not api_key:
        st.info("Paste your API key above to activate")
        st.stop()

# Choose model
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
    st.session_state.verdict_text = "No reading yet – ask your question above."

# ==============================
# NATURAL LANGUAGE → HYPEN TOPIC (the magic)
# ==============================
def sentence_to_topic(sentence):
    if not sentence.strip():
        return ""
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
                  'so', 'than', 'too', 'very', 'just', 'now'}
    # Find important words + numbers + South Africa etc.
    words = re.findall(r'\b(?:\d{4}|59|60|202[56789]|rand|south\s*africa|suid-afrika|[a-zA-Z]{4,})\b', sentence.lower())
    clean = [w.replace(' ', '').replace('southafrica', 'SouthAfrica').capitalize() for w in words if w not in stop_words]
    return "–".join(clean) if clean else "Unknown"

# ==============================
# PERSONAL VERDICT ENGINE
# ==============================
def generate_personal_verdict(topic, coherence, ratio, grid_df):
    try:
        dec_q = grid_df.loc["Decision Quantum", "Revelation"]
        blueprint = grid_df.loc["Blueprint / Soul", "Refinement"]
        creator_rev = grid_df.loc["Creator Layer", "Revelation"]
        continuity = grid_df.loc["Existence", "Continuity"]
    except:
        dec_q = blueprint = creator_rev = continuity = ""

    topic_lower = topic.lower()
    has_family = any(w in topic_lower for w in ["family", "legacy", "children", "wife", "obligation", "duty"])
    late_50s_sa = ("59" in topic_lower or "60" in topic_lower) and "south" in topic_lower

    verdict = []
    if coherence >= 98.0:
        verdict.append("Green light – maximum soul alignment.")
    elif coherence >= 96.0:
        verdict.append("Strong green light – this is your path.")
    elif coherence >= 94.0:
        verdict.append("Green light – go, but with the refinements below.")
    else:
        verdict.append("Caution – deeper clearing still required.")

    if ("self-employ" in topic_lower or "business" in topic_lower or "job" in topic_lower) and has_family and late_50s_sa:
        verdict.extend([
            "At 59–60 in South Africa the family responsibility is not a chain – it is rocket fuel.",
            "Build the venture so the family rides with you (consultancy that employs them, property/renewables as family asset, advisory practice using your lifetime expertise).",
            "Employment has become a lid. Self-employment that carries the family forward is the only move that rectifies decades of duty and secures legacy."
        ])

    strong = [s for s in [dec_q, blueprint, creator_rev, continuity] if isinstance(s, str) and len(s) > 20]
    if strong:
        verdict.append("\nKey transmissions from the heptagon:")
        verdict.extend(["→ " + s.strip() for s in strong[:3]])

    if abs(ratio - 3.741657) < 0.03:
        verdict.append("√14 lock → this is archetypally exact for you right now.")
    return "\n\n".join(verdict)

# ==============================
# CORE ANALYSIS
# ==============================
def analyse(topic):
    matrix = []
    with st.spinner(f"Running LOGOS analysis on **{topic}**…"):
        for row in matrix_questions:
            filled = []
            for q in row:
                prompt = f"Topic: {topic}\nQuestion: {q}\nAnswer in 8–15 deep words blending physics and metaphysics:"
                try:
                    ans = llm.invoke(prompt).content.strip()
                except:
                    ans = "…"
                filled.append(ans)
            matrix.append(filled)
    return np.array(matrix)

# ==============================
# UI
# ==============================

st.set_page_config(page_title="LOGOS Oracle", layout="wide")
st.title("LOGOS Heptagon Oracle")
st.markdown("Ask any real question – write it normally. The oracle understands.")

col1, col2 = st.columns([3, 1])
with col1:
    natural_sentence = st.text_input(
        "Your question (just write naturally)",
        placeholder="Should I start my own business at 59 with family duties in South Africa?",
        label_visibility="collapsed"
    )
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("Ask the Oracle", type="primary", use_container_width=True)

# Show what the oracle understood
topic = sentence_to_topic(natural_sentence)
if natural_sentence.strip() and topic and topic != "Unknown":
    st.caption(f"Interpreted as → **{topic}**")

# Quick presets
st.markdown("#### Or try one of these")
cols = st.columns(6)
for i, q in enumerate([
    "Will my marriage survive the betrayal?",
    "What does 2026 hold for South Africa?",
    "Should I forgive my father?",
    "Is my health scare the end?",
    "Start business or stay in job at 59?",
    "What is my soul’s true work?"
]):
    with cols[i % 6]:
        if st.button(q.split("?")[0][:20] + "…", use_container_width=True):
            natural_sentence = q
            topic = sentence_to_topic(q)
            run = True

# ==============================
# RUN & DISPLAY
# ==============================
if run and topic and topic != "Unknown":
    result = analyse(topic)
    df = pd.DataFrame(result, index=layers, columns=planes)

    total_chars = sum(len(str(c)) for row in result for c in row)
    avg_len = total_chars / 49
    coherence = round(min(avg_len * 2.7, 99.99), 2)
    ratio = round(avg_len / 10, 3)

    verdict = generate_personal_verdict(topic, coherence, ratio, df)
    verdict_text = f"LOGOS PERSONAL VERDICT\n{'='*60}\nTopic: {topic}\nQuestion: {natural_sentence}\nDate: {datetime.datetime.now():%Y-%m-%d %H:%M}\nCoherence: {coherence}% │ Ratio: {ratio}/1.000\n\n{verdict}"

    st.session_state.df = df
    st.session_state.coherence = coherence
    st.session_state.ratio = ratio
    st.session_state.topic = topic
    st.session_state.natural_sentence = natural_sentence
    st.session_state.verdict_text = verdict_text
    st.rerun()

# ==============================
# SHOW RESULTS
# ==============================
if st.session_state.df is not None:
    st.success(f"Oracle has spoken on **{st.session_state.topic}**")

    st.markdown(f"**Your question:** {st.session_state.natural_sentence}")
    st.markdown(f"**Resonance Coherence:** {st.session_state.coherence:.2f}% │ **Heptagonal Ratio:** {st.session_state.ratio:.3f}/1.000")

    if st.session_state.coherence >= 92.0:
        st.subheader("PERSONAL VERDICT")
        st.success(st.session_state.verdict_text.split("\n\n", 1)[1])
        st.markdown("---")

    st.dataframe(st.session_state.df.style.set_properties(**{'text-align': 'left'}), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        csv = st.session_state.df.to_csv().encode()
        st.download_button("Download Full Grid (CSV)", csv, f"LOGOS_{st.session_state.topic}.csv", "text/csv")
    with col2:
        st.download_button("Download Personal Verdict (TXT)", st.session_state.verdict_text.encode(),
                           f"VERDICT_{st.session_state.topic}.txt", "text/plain")
else:
    st.info("Ask any real question above – just write it like you would to a wise friend.")


