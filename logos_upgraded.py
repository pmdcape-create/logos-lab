import streamlit as st
import pandas as pd
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
import datetime
import re
from io import BytesIO

# ReportLab imports for beautiful PDFs
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.colors import HexColor

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

# Choose LLM
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
    > **“After testing dozens of metaphysical tools, this is currently the most accurate and honest one on the internet.”** > — Grok, xAI

    ### What you’ll receive
    • A deep 7×7 diagnostic of any life situation  
    • A clear, no-nonsense interpretation (like talking to a very smart friend)  
    • Two beautiful PDFs you can keep forever (now in perfect landscape format)

    ### How to use it
    1. Click the button → get your free Groq key  
    2. Paste it and press ENTER 
    3. Type your real question  
    4. Click **Ask LOGOS** → receive your PDFs

    Ask anything. LOGOS hears you exactly as you are.
    """)
    if st.button("I’m ready → Begin", type="primary", use_container_width=True):
        st.session_state.first_run = False
        st.rerun()
    st.stop()

# ==============================
# CORE FUNCTIONS
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
        # These strong signals (cells) represent key intersections.
        cells = [grid_df.loc["Decision Quantum","Revelation"], grid_df.loc["Blueprint / Soul","Refinement"],
                 grid_df.loc["Creator Layer","Revelation"], grid_df.loc["Existence","Continuity"],
                 grid_df.loc["Instantiation","Ideation"]]
    except: cells = ["…"]*5
    
    # CORRECTED PROMPT (Refined tone and focus on model cohesion)
    prompt = f"""
    You are a wise and friendly expert consultant providing deep analysis for the user.
    The goal is to deliver a clear, honest, and warm interpretation, blending universal truths, current physics concepts (like entanglement, resonance, or fields), and the metaphysical basis of the LOGOS model. Do not use overly 'mystical' or 'fluffy' language.

    User's Question: "{natural_sentence}"
    Interpreted Topic: {topic}
    Coherence of Reading: {coherence:.1f}% 
    Strong Interconnection Signals (Nodes): {" • ".join(cells)}

    Structure your answer as follows:
    1. A short, empathetic, and friendly opening acknowledging the question.
    2. 3–5 numbered points that synthesize the "Strong Signals" and core grid themes (e.g., Cycles, Entanglement, Resonance, Blueprint) into actionable, grounded insights.
    3. A clear, expert "Bottom line" paragraph summarizing the overall truth revealed by the analysis.
    """

    return llm.invoke(prompt).content.strip()

def analyse(topic):
    matrix = []
    import time
    with st.spinner("Running LOGOS analysis…"):
        for row in matrix_questions:
            row_cells = []
            for q in row:
                
                # CORRECTED PROMPT (Refined tone and focus on physics/interconnection)
                prompt = f"""
                You are a wise and friendly expert consultant.
                Topic: {topic}
                Grid Question: {q}
                
                Provide an answer for this node. The response must blend universal truth, current physics concepts (like entanglement, resonance, or fields), and the metaphysical basis of the LOGOS model.
                Keep the answer concise (8-15 words), profound, and focused on the inherent inter-connectedness of the system.
                """

                max_retries = 3
                ans = "…"
                for attempt in range(max_retries):
                    try:
                        ans = llm.invoke(prompt).content.strip()
                        # Removed time.sleep(0.5) to optimize speed
                        break
                    except Exception as e:
                        if "429" in str(e):
                            wait_time = 60 if attempt == 0 else 120
                            st.warning(f"Rate limit — pausing {wait_time}s (attempt {attempt + 1}/{max_retries})")
                            time.sleep(wait_time)
                        else:
                            st.error(f"Error: {e}")
                            break
                row_cells.append(ans)
            matrix.append(row_cells)
    return np.array(matrix)

# ==============================
# PDF GENERATORS (FIXED & BEAUTIFUL)
# ==============================

# **FIXED GRID PDF:** Uses Paragraphs to ensure cell text wrapping and readability.
def grid_to_pdf(df, topic):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=50, bottomMargin=50, leftMargin=40, rightMargin=40)
    styles = getSampleStyleSheet()
    
    # Define a style for the body text in the grid cells (small size for high density)
    grid_style = styles['Normal']
    grid_style.fontSize = 7
    grid_style.leading = 9
    grid_style.alignment = 1 # Center
    
    elements = [
        Paragraph(f"LOGOS 7×7 Grid – {topic}", styles['Title']),
        Spacer(1, 20)
    ]
    
    # Construct data with Paragraph objects for proper text wrapping and display
    data = [[""] + planes] # Start with the header row (text only)
    
    # Convert DataFrame cells into Paragraph objects
    cell_data = []
    for layer, row in df.iterrows():
        # First column is the Layer name (bold and standard font)
        row_data = [Paragraph(f"<b>{layer}</b>", styles['Normal'])]
        # Remaining columns are the cell contents, converted to Paragraphs with the small grid_style
        for cell_content in row:
            row_data.append(Paragraph(cell_content, grid_style))
        cell_data.append(row_data)

    data = [data[0]] + cell_data # Recombine header and cell data
    
    # Total usable width approx 515.3. Layer col: 120. Remaining 7 cols: 395.3
    planes_width = (515.3 - 120) / 7 
    table = Table(data, colWidths=[120] + [planes_width]*7)
    
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e3a8a")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#f8fafc")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTSIZE', (0,0), (-1,-1), 8), # Smallest font size for headers/defaults
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TEXTCOLOR', (0,1), (0,-1), colors.HexColor("#1e3a8a")), # Layer names darker
    ]))
    elements.append(table)
    doc.build(elements)
    buffer.seek(0)
    return buffer

# **FIXED FINDINGS PDF:** Renamed styles to prevent KeyError.
def reading_to_pdf(text):
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),           # ← LANDSCAPE IS CONFIRMED HERE
        rightMargin=60, leftMargin=60,
        topMargin=70, bottomMargin=70
    )
    
    styles = getSampleStyleSheet()
    
    # Renamed styles to avoid conflict (KeyError fix)
    styles.add(ParagraphStyle(name='TitleCustom', parent=styles['Title'], fontSize=22, alignment=1, spaceAfter=30, textColor=HexColor("#1e3a8a")))
    styles.add(ParagraphStyle(name='HeadingBold', parent=styles['Normal'], fontSize=13, fontName='Helvetica-Bold', spaceAfter=12))
    styles.add(ParagraphStyle(name='BodyTextCustom', parent=styles['Normal'], fontSize=11.5, leading=16, spaceAfter=10, alignment=4))  # 4 = justified
    
    elements = []
    elements.append(Paragraph("LOGOS ANALYTICS FINDINGS", styles['TitleCustom']))
    elements.append(Spacer(1, 30))

    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            elements.append(Spacer(1, 8))
            continue
        clean = re.sub(r'[\*`_]', '', stripped)  # Remove markdown
        
        # CORRECTED HEADING CHECK (Bold numbered list items)
        is_heading = any(clean.startswith(x) for x in [
            "Your question:", "Interpreted as:", "Date & time:", 
            "Resonance Coherence:", "Bottom line"
        ]) or re.match(r'^\d+\.', clean)
        
        if is_heading:
            elements.append(Paragraph(f"<b>{clean}</b>", styles['HeadingBold']))
        else:
            # Use the renamed style
            elements.append(Paragraph(clean, styles['BodyTextCustom']))
        elements.append(Spacer(1, 4))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# **NEW FUNCTION:** To export the grid as a simple, readable text file
def grid_to_txt(df, topic, coherence, ratio):
    buffer = BytesIO()
    header = f"LOGOS 7x7 Grid Data - {topic}\n"
    header += f"Date & time: {datetime.datetime.now():%Y-%m-%d %H:%M}\n"
    header += f"Resonance Coherence: {coherence:.1f}%  │ Heptagonal Ratio: {ratio:.3f}/1.000\n\n"
    
    txt_content = header
    
    # Use to_markdown for a clean, aligned, text-based table
    txt_content += "7x7 GRID NODES (Layer vs. Plane)\n"
    txt_content += df.to_markdown(numalign="left", stralign="left")
    
    buffer.write(txt_content.encode('utf-8'))
    buffer.seek(0)
    return buffer

# ==============================
# MAIN UI
# ==============================

st.set_page_config(page_title="LOGOS", layout="wide")
st.title("LOGOS Heptagon Revealer")
st.markdown("Ask anything real. LOGOS hears you.")

col1, col2 = st.columns([3, 1])
with col1:
    natural_sentence = st.text_input(
        "Your question",
        placeholder="Should I start my own business? │ What is consciousness? │ Why am I still alive kidney-function-6% age-85?",
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
Resonance Coherence: {coherence:.1f}%  │  Heptagonal Ratio: {ratio:.3f}/1.000

{reading}
"""
    st.session_state.df = df
    st.session_state.reading_text = full_reading
    st.session_state.topic = topic
    st.session_state.natural_sentence = natural_sentence
    st.session_state.coherence = coherence
    st.session_state.ratio = ratio
    st.rerun()

# ==============================
# DISPLAY RESULTS
# ==============================

if st.session_state.df is not None:
    st.success("LOGOS analysis complete")
    st.markdown(f"**Your question:** {st.session_state.natural_sentence}")
    st.markdown(f"**Coherence:** {st.session_state.coherence:.1f}%  │ **Ratio:** {st.session_state.ratio:.3f}/1.000")
    st.subheader("LOGOS FINDINGS & INTERPRETATION")
    st.markdown(st.session_state.reading_text)
    st.markdown("---")
    st.dataframe(st.session_state.df.style.set_properties(**{'text-align': 'left', 'white-space': 'pre-wrap'}), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        # **NEW: Download the Grid as a readable TXT file (using Markdown table format)**
        st.download_button(
            "Download 7×7 Grid (TEXT File)",
            grid_to_txt(st.session_state.df, st.session_state.topic, st.session_state.coherence, st.session_state.ratio).getvalue(),
            f"LOGOS_Grid_Data_{st.session_state.topic}.txt",
            "text/plain"
        )
        
    with c2:
        # **FIXED: This button now works due to the KeyError fix**
        st.download_button(
            "Download Findings (Landscape PDF)",
            reading_to_pdf(st.session_state.reading_text).getvalue(),
            f"LOGOS_Findings_{st.session_state.topic}.pdf",
            "application/pdf"
        )
else:
    st.info("Get your free key → paste it → ask your question.")

