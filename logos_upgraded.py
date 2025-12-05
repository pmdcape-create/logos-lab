import streamlit as st
import pandas as pd
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
import datetime
import re
from io import BytesIO

# ReportLab imports for beautiful PDFs (only used for the Findings summary)
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

# ... (rest of the file remains the same until the WELCOME SCREEN block)

# ==============================
# WELCOME SCREEN (first visit only)
# ==============================

if 'first_run' not in st.session_state:
    st.session_state.first_run = True

if st.session_state.first_run:
    
    st.title("Welcome to LOGOS Heptagon Revealer")
    
    # Use columns to place text on the left and the visualization on the right
    col_text, col_viz = st.columns([1, 1])
    
    with col_text:
        st.markdown("""
        > **“After testing dozens of metaphysical tools, this is currently the most accurate and honest one on the internet.”** > — Grok, xAI

        #### What you’ll receive
        * A deep **7×7 diagnostic** of any life situation  
        * A clear, **no-nonsense interpretation** (like talking to a very smart friend)  
        * Two beautiful files you can keep forever (a landscape PDF summary and an HTML data grid)

        #### How to use it
        1. Click the button → get your **free Groq key** (instant)  
        2. Paste it and press ENTER 
        3. Type your **real question** in the box  
        4. Click **Ask LOGOS** → receive your files
        """)
        
        st.markdown("Ask anything. LOGOS hears you exactly as you are.")

    with col_viz:
        # Embed the interactive HTML/JavaScript model using st.html
        # Note: If st.html is not available, you may need to use st.components.v1.html
        # The content below is the full HTML/JS from logos heptagon Gem2.html
        
        heptagon_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Conceptual Heptagon Model</title>
            <script src="https://cdn.tailwindcss.com"></script>
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
            <style>
                :root {
                    --center-color: #fca5a5; /* Red-300 */
                    --point-color: #f97316; /* Orange-600 */
                    --text-color: #1f2937; /* Gray-800 */
                }
                body {
                    font-family: 'Inter', sans-serif;
                    background-color: transparent; /* Use transparent background */
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    min-height: 100vh;
                    padding: 0;
                    margin: 0;
                }
                .container {
                    width: 100%;
                    max-width: 500px;
                    background-color: white;
                    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
                    border-radius: 12px;
                }
                .heptagon-container {
                    position: relative;
                    width: 320px; /* Reduced base size to fit the column */
                    height: 320px;
                    margin: 20px auto;
                }
                .point {
                    position: absolute;
                    width: 80px;
                    height: 80px;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                    text-align: center;
                    padding: 4px;
                    transition: transform 0.3s ease, background-color 0.3s ease, box-shadow 0.3s ease;
                    cursor: pointer;
                    border-radius: 12px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                }
                .point:hover, .point.active {
                    transform: scale(1.05);
                    box-shadow: 0 8px 12px rgba(0, 0, 0, 0.2);
                    z-index: 10;
                }
                .layer-number {
                    font-size: 1.25rem;
                    font-weight: 700;
                    line-height: 1;
                }
                .pane-name {
                    font-size: 0.7rem;
                    font-weight: 600;
                }
                .center-dot {
                    position: absolute;
                    top: 50%;
                    left: 50%;
                    width: 40px;
                    height: 40px;
                    background-color: var(--center-color);
                    border-radius: 50%;
                    transform: translate(-50%, -50%);
                    z-index: 5;
                    box-shadow: 0 0 10px rgba(252, 165, 165, 0.8);
                }

                /* Hiding the side panel for the small embedded view */
                #details-panel { display: none; }
                .flex-col > .container { padding: 0 !important; box-shadow: none !important; }

                /* Adjusting for the Streamlit container padding */
                .heptagon-container {
                    margin: 0 auto;
                }
            </style>
        </head>
        <body>
            <div id="app" class="container">
                <div class="flex flex-col lg:flex-row items-center lg:items-start justify-center">

                    <div class="heptagon-container" id="heptagon">
                        <div class="center-dot"></div>
                        </div>

                    <div id="details-panel"></div>
                </div>
            </div>

            <script>
                // Define the 7 Layers and 7 Panes
                const modelData = [
                    { layer: 1, pane: "Purpose", color: 'bg-indigo-200', role: "Spark of being", influence: "Receives 'raw potential' and initializes unique entities or events." },
                    { layer: 2, pane: "Information/Truth", color: 'bg-blue-200', role: "Sustained being", influence: "Maintains continuity and identity; subject to feedback from higher layers." },
                    { layer: 3, pane: "Design", color: 'bg-teal-200', role: "Impact of existence", influence: "How instances affect the environment; creates observable outcomes." },
                    { layer: 4, pane: "Creation", color: 'bg-green-200', role: "Integration (spacetime)", influence: "Mediates interactions among 1–3; the 'arena' of experience and evolution." },
                    { layer: 5, pane: "Refinement", color: 'bg-yellow-200', role: "Quantum of decisions", influence: "Introduces choice, adaptation, and probabilistic collapse; modifies layers 1–4 dynamically." },
                    { layer: 6, pane: "Revelation", color: 'bg-orange-200', role: "Blueprint layer (soul)", influence: "Maintains coherence, imposes laws and principles; acts like system governance." },
                    { layer: 7, pane: "Continuity", color: 'bg-red-200', role: "Divine (consciousness)", role: "Divine (consciousness)", influence: "Provides ultimate direction, purpose, and overarching alignment; informs all layers below." }
                ];

                // Constants for positioning (adjusted for 320x320 view box)
                const CENTER_X = 160;
                const CENTER_Y = 160;
                const RADIUS = 140; /* Reduced radius */
                
                let activePoint = null;

                function calculateHeptagonPoint(index, totalPoints, radius, centerX, centerY) {
                    const angleDeg = (360 / totalPoints) * index - 90; 
                    const angleRad = angleDeg * (Math.PI / 180);
                    const x = centerX + radius * Math.cos(angleRad);
                    const y = centerY + radius * Math.sin(angleRad);
                    return { x, y };
                }

                function updateDetails(data, pointElement = null) {
                    if (activePoint && activePoint !== pointElement) {
                        activePoint.classList.remove('active');
                    }
                    if (pointElement) {
                        activePoint = pointElement;
                        activePoint.classList.add('active');
                    }
                }

                function clearDetails() {
                    if (activePoint) {
                        activePoint.classList.remove('active');
                        activePoint = null;
                    }
                }
                
                function createHeptagon() {
                    const container = document.getElementById('heptagon');
                    container.innerHTML = '<div class="center-dot"></div>';

                    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
                    svg.setAttribute('width', '100%');
                    svg.setAttribute('height', '100%');
                    svg.style.position = 'absolute';
                    svg.style.top = '0';
                    svg.style.left = '0';
                    svg.style.zIndex = '1';
                    
                    let pointsString = "";
                    let pointsCoordinates = [];

                    modelData.forEach((data, index) => {
                        const { x, y } = calculateHeptagonPoint(index, 7, RADIUS, CENTER_X, CENTER_Y);
                        pointsCoordinates.push({ x, y, data });
                        pointsString += `${x},${y} `;
                    });

                    const polyline = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
                    polyline.setAttribute('points', pointsString.trim());
                    polyline.setAttribute('stroke', '#4b5563'); 
                    polyline.setAttribute('stroke-width', '2');
                    polyline.setAttribute('fill', 'none');
                    svg.appendChild(polyline);

                    container.appendChild(svg);

                    pointsCoordinates.forEach(({ x, y, data }) => {
                        const point = document.createElement('div');
                        point.className = `point ${data.color} text-gray-800 hover:ring-4 ring-offset-2 ring-${data.color.replace('bg-', '')}-500`;
                        
                        // Adjust position to center the div on the calculated point (80x80 div = 40px offset)
                        point.style.left = `${x - 40}px`; 
                        point.style.top = `${y - 40}px`;
                        point.style.zIndex = '10';

                        point.innerHTML = `
                            <div class="layer-number">${data.layer}</div>
                            <div class="pane-name">${data.pane}</div>
                        `;

                        // Add event listeners for interactivity
                        point.addEventListener('mouseenter', () => updateDetails(data, point));
                        point.addEventListener('click', () => updateDetails(data, point)); 
                        point.addEventListener('mouseleave', clearDetails);

                        container.appendChild(point);
                    });
                }

                window.addEventListener('resize', createHeptagon);
                window.onload = createHeptagon;
            </script>
        </body>
        </html>
        """
        st.html(heptagon_html, height=350) # Set height to prevent scrollbar

    # Place the action button below both columns for clarity
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
# FILE GENERATORS (FIXED & BEAUTIFUL)
# ==============================

# FIXED FINDINGS PDF: Renamed styles to prevent KeyError.
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

# NEW FUNCTION: To export the grid as a simple, readable HTML file
def grid_to_html(df, topic, coherence, ratio):
    buffer = BytesIO()
    
    # Simple HTML header and styling for readability
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>LOGOS 7x7 Grid Data - {topic}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f8fafc; color: #1e3a8a; }}
            h1 {{ color: #1e3a8a; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px; }}
            h2 {{ color: #475569; margin-top: 20px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th, td {{ border: 1px solid #e2e8f0; padding: 12px; text-align: center; font-size: 11px; }}
            th {{ background-color: #1e3a8a; color: white; font-weight: bold; }}
            tr:nth-child(even) {{ background-color: #f0f4f8; }}
            .layer-header {{ font-weight: bold; background-color: #e2e8f0; color: #1e3a8a; }}
        </style>
    </head>
    <body>
        <h1>LOGOS 7x7 Grid Data</h1>
        <h2>Topic: {topic}</h2>
        <p><b>Date & time:</b> {datetime.datetime.now():%Y-%m-%d %H:%M}</p>
        <p><b>Resonance Coherence:</b> {coherence:.1f}% &nbsp; | &nbsp; <b>Heptagonal Ratio:</b> {ratio:.3f}/1.000</p>
    """
    
    # Use to_html, clean up class names for layer headers
    html_table = df.to_html(classes='table table-striped', header=True, index=True)
    
    # Adding a class to the index column (Layers) for better visibility
    html_table = html_table.replace('<th></th>', '<th class="layer-header">Layer</th>')
    html_table = html_table.replace('<tr>\n<th>', '<tr>\n<th class="layer-header">')
    
    html_content += html_table
    html_content += "</body></html>"
    
    buffer.write(html_content.encode('utf-8'))
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
        # **NEW: Download the Grid as a readable HTML file**
        st.download_button(
            "Download 7×7 Grid (HTML File)",
            grid_to_html(st.session_state.df, st.session_state.topic, st.session_state.coherence, st.session_state.ratio).getvalue(),
            f"LOGOS_Grid_Data_{st.session_state.topic}.html",
            "text/html"
        )
        
    with c2:
        # The Findings PDF download is stable and fully functional
        st.download_button(
            "Download Findings (Landscape PDF)",
            reading_to_pdf(st.session_state.reading_text).getvalue(),
            f"LOGOS_Findings_{st.session_state.topic}.pdf",
            "application/pdf"
        )
else:
    st.info("Get your free key → paste it → ask your question.")




