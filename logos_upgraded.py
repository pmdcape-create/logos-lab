import streamlit as st
import pandas as pd
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

# ==============================
# DATA
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
# SIDEBAR – CLEAN API KEY
# ==============================

with st.sidebar:
    st.header("LOGOS Engine")
    api_key = st.text_input("OpenAI or Groq API key", type="password", help="Free Groq key → https://console.groq.com/keys")
    if not api_key:
        st.info("Paste your API key above to activate the heptagon")
        st.stop()

# Choose LLM
if api_key.startswith("gsk_"):
    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key, temperature=0.7)
else:
    llm = ChatOpenAI(model="gpt-4o", api_key=api_key, temperature=0.7)

# ==============================
# CORE FUNCTION – NO CACHING
# ==============================

def analyse(topic):
    matrix = []
    with st.spinner(f"Analysing **{topic}** across all 49 nodes…"):
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
# APP
# ==============================

st.set_page_config(page_title="LOGOS Heptagon Lab", layout="wide")
st.title("LOGOS Model Logic Laboratory")
st.markdown("Enter any concept and watch the 7×7 heptagon reveal its eternal pattern")

col1, col2 = st.columns([3,1])
with col1:
    topic = st.text_input("Topic", placeholder="Gravity · Love · Entropy · Free Will · Light", label_visibility="collapsed")
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("Run Analysis", type="primary", use_container_width=True)

# One-click sacred topics
st.markdown("#### Sacred presets:")
cols = st.columns(6)
presets = ["Gravity", "Love", "Free Will", "Entropy", "Light", "Forgiveness"]
for i, name in enumerate(presets):
    with cols[i % 6]:
        if st.button(name, use_container_width=True):
            topic = name
            run = True

if run and topic:
    result = analyse(topic)
    df = pd.DataFrame(result, index=layers, columns=planes)
    
    st.success(f"Complete LOGOS analysis of **{topic}**")
    st.dataframe(df, use_container_width=True)
    
    # Simple resonance score
    coherence = round(sum(len(c) for row in result for c in row) / len(result.flatten()) / 25, 3)
    st.metric("Resonance Coherence", f"{coherence:.3f}/1.000")
    
    csv = df.to_csv().encode()
    st.download_button("Download CSV", csv, f"LOGOS_{topic}.csv", "text/csv")

import streamlit as st

# ──────────────────────────────────────────────────────────────
# ADD THIS FUNCTION ANYWHERE IN YOUR MAIN FILE (after imports)
# ──────────────────────────────────────────────────────────────
def generate_personal_verdict(topic: str, coherence: float, ratio: float, grid_df):
    """
    Turns a high-coherence grid into a short, brutally clear personal verdict.
    """
    # Extract the hottest cells (these are almost always the carriers of the real message)
    try:
        dec_q = grid_df.loc["Decision Quantum", "Revelation"]
        blueprint = grid_df.loc["Blueprint / Soul", "Refinement"]
        creator_rev = grid_df.loc["Creator Layer", "Revelation"]
        continuity = grid_df.loc["Existence", "Continuity"]
    except:
        dec_q = blueprint = creator_rev = continuity = ""

    # Detect some common high-signal SA / late-50s patterns
    topic_lower = topic.lower()
    is_sa_59 = all(x in topic_lower for x in ["south", "59", "man", "family", "obligation", "legacy"])
    has_family = any(word in topic_lower for word in ["family", "legacy", "children", "wife", "obligation"])

    # Base verdict
    verdict = []

    if coherence >= 98.0:
        verdict.append("Green light – maximum soul alignment.")
    elif coherence >= 96.0:
        verdict.append("Strong green light – this is your path.")
    elif coherence >= 94.0:
        verdict.append("Green light – go, but with the refinements below.")
    else:
        verdict.append("Caution – deeper clearing still required.")

    if "self-employ" in topic_lower or "entrepreneur" in topic_lower:
        if has_family and is_sa_59:
            verdict.append("At 59 in South Africa the family obligation is not a chain – it is rocket fuel.")
            verdict.append("Build the business so the family rides with you (consultancy that employs them, property/renewables that become the new asset base, advisory practice using your lifetime expertise).")
            verdict.append("Traditional employment has become a lid. Self-employment that carries the family forward is the only move that rectifies decades of duty and secures legacy beyond your lifetime.")
        else:
            verdict.append("Leap. The container of employment is complete for this lifetime.")

    # Add any direct Revelation/Continuity quotes that feel like punches
    strong_lines = []
    for line in [dec_q, blueprint, creator_rev, continuity]:
        if isinstance(line, str) and len(line) > 20:
            strong_lines.append(f"“{line.strip()}”")

    if strong_lines:
        verdict.append("Key transmissions from the grid:")
        verdict.extend(["→ " + l for l in strong_lines[:3]])

    # Final ratio note
    if abs(ratio - 3.741657) < 0.015:
        verdict.append("√14 lock → this is archetypally exact for you right now.")
    elif abs(ratio - 3.5) < 0.02:
        verdict.append("7/2 lock → perfect half-heptad harmony.")

    return "\n\n".join(verdict)


# ──────────────────────────────────────────────────────────────
# REPLACE YOUR CURRENT GRID DISPLAY SECTION WITH THIS
# ──────────────────────────────────────────────────────────────
# (assuming you already have variables: topic, coherence, ratio, df)

st.markdown(f"**Topic:** {topic}")
st.markdown(f"**Resonance Coherence:** {coherence:.2f} %  |  **Heptagonal Ratio:** {ratio:.3f}/1.000")

# <<< THE NEW PART STARTS HERE >>>
if coherence >= 92.0:                     # you can raise this threshold if you want
    verdict = generate_personal_verdict(topic, coherence, ratio, df)
    st.subheader("PERSONAL VERDICT")
    st.success(verdict)
    st.markdown("---")

# Original table display (unchanged)
st.dataframe(df.style.set_properties(**{'text-align': 'left'}))



