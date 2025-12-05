import streamlit as st
import pandas as pd
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
import datetime
import re
import json
from io import BytesIO

# NEW: Import for the Heptagon visualization
import streamlit.components.v1 as components

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

# Note: matrix_questions is included for historical context but is superseded by the new
# LLM prompt which requests a full matrix and is guided by the LAYER_GUIDING_QUESTIONS.
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
# LOGOS SYSTEMIC CONSTRAINTS (NEWLY ADDED)
# These are used to constrain the LLM's philosophical and logical output.
# ==============================

# 1. The 10 Non-Negotiable Laws
LOGOS_LAWS_TEXT = """
1. Law of Aligned Instantiation: Purpose is defined at Pane 1 and embedded into the instantiated form. Misalignment causes rejection or failure of form.
2. Law of Quantum Instantiation: Decision-making mirrors quantum collapse of possibility into actuality. Only one path is taken — all others collapse.
3. Law of Systemic Feedback & Regulation: The system adjusts based on outcomes — patterns that violate alignment are pruned through consequences or failure.
4. Law of Consistency & Sustainability: Sustainable instantiations preserve energetic and logical integrity. Inconsistency leads to breakdown over time.
5. Law of Guidance via Alignment with Higher Order: Guidance flows from higher-order structures into decisions — the soul/conscience‘listens’ through systemic resonance.
6. Law of Sustained Continuity: Continuity is earned, not guaranteed. Only those instantiations that fit the whole system’s telos are preserved.
7. Law of Telic Resolution: Every part of the system exists for a defined purpose. When misaligned, the system seeks to repair or reject the part. Purpose is not optional — it is the condition for continued existence.
8. Law of Iterative Collapse: The act of choosing transforms potential into form. Decision-making is the engine of emergence. Each iteration locks in a configuration.
9. Law of Bidirectional Blueprint Influence: The soul/conscienceprovides directional purpose; the mind interprets it and acts within the embodied field. Consciousness emerges from this dialogue.
10. Law of Deferral-Resolution: Decisions are not optional. Initial observation is a decision; all changes must be accepted or rejected. Deferrals create instability and will be re-presented by the system until resolved. This sustains function and alignment.
"""

# 2. The Guiding Questions to enforce internal reasoning (from Logic Qs.csv)
LAYER_GUIDING_QUESTIONS = """
1. Instantiation: What sparked this idea or insight?
2. Existence: How does this idea manifest in lived reality?
3. Effect / Impact: What are the consequences or effects?
4. Iteration: How does this repeat, evolve, or cycle?
5. Decision Quantum: Where is free will or choice present?
6. Blueprint / Soul: What archetypal pattern or law does this reflect?
7. Creator Layer: What does this reveal about source, origin, or divinity?
"""

# ==============================
# CORE HELPER FUNCTIONS
# ==============================

def sentence_to_topic(sentence):
    # This function remains unchanged for topic generation
    if not sentence.strip(): return ""
    numbers = re.findall(r'\b\d+%|\b\d+\b', sentence)
    numbers_clean = [n.replace('%','Percent') if '%' in n else f"Age{n}" if int(n)>=30 else n for n in numbers]
    stop_words = {'i','me','my','we','you','he','she','it','they','the','a','an','and','but','if','or','what','when','how','will','should','can','just','now','please'}
    words = re.findall(r'\b[a-zA-Z]{4,}\b', sentence.lower())
    clean_words = [w.capitalize() for w in words if w not in stop_words]
    parts = numbers_clean + clean_words
    seen = set(); unique = [p for p in parts if not (p in seen or seen.add(p))]
    return "–".join(unique) if unique else "Unknown"

def extract_json(text):
    # Extracts the first JSON object from the text.
    match = re.search(r'```json\s*(\{.*\})\s*```', text, re.DOTALL)
    if match:
        return match.group(1)
    return None

# ==============================
# AI GENERATION FUNCTION (MODIFIED AND UNIFIED)
# The previous `analyse` and `generate_structured_reading` functions are replaced by this single, constrained function.
# ==============================

@st.cache_data(show_spinner="Running LOGOS analysis...")
def generate_reading(natural_sentence, planes, layers, groq_key, openai_key):

    # 1. Initialize the LLM
    if groq_key and groq_key.startswith("gsk_"):
        # Using the user's preferred model (llama-3.3-70b-versatile) for Groq, adjusting temperature
        llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_key, temperature=0.3) 
    elif openai_key:
        # Using the user's preferred model (gpt-4o) for OpenAI, adjusting temperature
        llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_key, temperature=0.3)
    else:
        # st.error("Please enter a valid API key to proceed.")
        return None, None, None, None, None

    # 2. Construct the full system prompt with new constraints
    
    skeptic_filter_and_laws = f"""
    ## CRITICAL PREMISE FILTER (NON-NEGOTIABLE FIRST STEP)
    FIRST STEP: Before analysis, critically evaluate the user's input. If the question contains an unproven, supernatural, or scientifically non-neutral claim (e.g., 'ghost,' 'curse,' 'magic,' 'destiny'), you MUST NOT affirm or deny its literal existence. Instead, you must state that the analysis will proceed on two parallel tracks:
    1. The literal interpretation of the felt experience (e.g., 'the energy or sensation of being poked').
    2. The exploration of the experience as a metaphor, psychological projection, or cognitive phenomenon.
    This ensures methodological neutrality and rigor in line with current physics and the model's intent to track 'metaphysical probabilities.'

    ## LOGOS SYSTEMIC LAWS (RIGOROUS CONSTRAINT)
    YOUR ANALYSIS MUST STRICTLY ADHERE TO THE FOLLOWING 10 LOGOS SYSTEMIC LAWS. These laws supersede general helpfulness and must inform your interpretation of coherence, alignment, and consequence. If a user's premise or analysis violates a law, you must reference the violated law (e.g., 'This violates the Law of Telic Resolution').

    LOGOS LAWS:
    {LOGOS_LAWS_TEXT}

    ## LAYER-SPECIFIC GUIDING QUESTIONS (INTERNAL LOGIC ENFORCEMENT)
    When populating the matrix, you MUST use the corresponding Layer's Guiding Question as the primary lens for your response, ensuring logical integrity across the 'layers' (rows). The questions guide the core analysis of the row.

    GUIDING QUESTIONS (Layer: Core Question):
    {LAYER_GUIDING_QUESTIONS}
    """
    
    # 3. Final Prompt Structure
    full_prompt = f"""
    {skeptic_filter_and_laws}
    
    You are a LOGOS Analytical Engine. Your purpose is to conduct a 7x7 matrix analysis on a user's question, strictly following the defined methodology, the LOGOS Laws, and the Layer-Specific Guiding Questions provided above.

    The 7 vertical columns are called 'Panes' (or Planes): {planes}.
    The 7 horizontal rows are called 'Layers': {layers}.

    Your task is to analyze the user's sentence: "{natural_sentence}"

    **PROCESS:**
    1. **Matrix Population:** Fill the 7x7 grid. Each cell is the intersection of a Layer (Row) and a Plane (Column). The content must be a concise, analytical insight (8-15 words) reflecting the intersection, strictly informed by the Layer's Guiding Question and the LOGOS Laws.

    2. **Coherence & Ratio Calculation:**
        - **Coherence:** Determine the internal logical consistency of the topic (0-100%). A high score means the idea is aligned with the LOGOS Laws, has clear purpose (Telos), and sustains itself.
        - **Ratio:** Calculate the ratio of **Aligned/Generative** entries to **Misaligned/Reactive** entries (expressed as a decimal from 0.000 to 1.000). Entries that support clear purpose, health, and alignment are Aligned. Entries that indicate friction, reaction, or systemic failure are Misaligned.

    3. **Full Reading:** Write a detailed, clear, and warm interpretation. This narrative must seamlessly weave together the following sections:
        - **INTRODUCTION:** Acknowledge the core question and state the analytical framework (the two-track approach if the Critical Premise Filter was triggered, otherwise state the LOGOS framework).
        - **CORE INSIGHT (THE LOGOS):** State the main discovery and the coherence score/ratio.
        - **SYSTEMIC ANALYSIS (Law Enforcement):** Specifically reference **at least two** LOGOS Laws that are most relevant to the question's challenge or resolution. Explain how the question's circumstances either violate or align with these laws. This is essential for preventing random AI output.
        - **PANEL-BY-PANEL SUMMARY:** Briefly summarize the findings across the 7 Panes.
        - **CONCLUSION & RESOLUTION:** Offer a clear, actionable path for resolution based on the analysis, blending universal truths and current physics as intended.

    **OUTPUT FORMAT:**
    Your response must be a single block of text containing two distinct sections:
    
    1. **The JSON block:** This must contain only the structured data for the application.
    2. **The Full Reading block:** This is the detailed text for the user.
    
    Provide the JSON first, and then the Full Reading.
    
    ```json
    {{
        "topic": "A concise, analytical title for the question (e.g., Spiritual-Physical Entanglement)",
        "coherence": 85.0,
        "ratio": 0.785,
        "matrix": [
            ["Cell 1x1 Content...", "Cell 1x2 Content...", "Cell 1x3 Content...", "Cell 1x4 Content...", "Cell 1x5 Content...", "Cell 1x6 Content...", "Cell 1x7 Content..."],
            ["Cell 2x1 Content...", "Cell 2x2 Content...", "Cell 2x3 Content...", "Cell 2x4 Content...", "Cell 2x5 Content...", "Cell 2x6 Content...", "Cell 2x7 Content..."],
            ["Cell 3x1 Content...", "Cell 3x2 Content...", "Cell 3x3 Content...", "Cell 3x4 Content...", "Cell 3x5 Content...", "Cell 3x6 Content...", "Cell 3x7 Content..."],
            ["Cell 4x1 Content...", "Cell 4x2 Content...", "Cell 4x3 Content...", "Cell 4x4 Content...", "Cell 4x5 Content...", "Cell 4x6 Content...", "Cell 4x7 Content..."],
            ["Cell 5x1 Content...", "Cell 5x2 Content...", "Cell 5x3 Content...", "Cell 5x4 Content...", "Cell 5x5 Content...", "Cell 5x6 Content...", "Cell 5x7 Content..."],
            ["Cell 6x1 Content...", "Cell 6x2 Content...", "Cell 6x3 Content...", "Cell 6x4 Content...", "Cell 6x5 Content...", "Cell 6x6 Content...", "Cell 6x7 Content..."],
            ["Cell 7x1 Content...", "Cell 7x2 Content...", "Cell 7x3 Content...", "Cell 7x4 Content...", "Cell 7x5 Content...", "Cell 7x6 Content...", "Cell 7x7 Content..."]
        ]
    }}
    ```

    <FULL_READING_START>
    **LOGOS Analysis for [Topic Title]**
    ... [Detailed interpretive text follows, structured as requested] ...
    <FULL_READING_END>
    """
    
    # 4. Call the LLM
    try:
        response = llm.invoke(full_prompt).content

        json_text = extract_json(response)
        if not json_text:
             st.error("AI response error: Could not extract valid JSON structure from the LLM. Please try again.")
             return None, None, None, None, None
             
        data = json.loads(json_text)
        
        # Get the full reading text
        reading_start_tag = "<FULL_READING_START>"
        reading_end_tag = "<FULL_READING_END>"
        
        start_index = response.find(reading_start_tag) + len(reading_start_tag)
        end_index = response.find(reading_end_tag)
        
        if end_index > start_index and start_index != -1 and end_index != -1:
            full_reading = response[start_index:end_index].strip()
        else:
             # Fallback if tags are missed
             full_reading = "Error: Could not parse the full reading narrative. Data was received."


        df_data = data["matrix"]
        topic = data["topic"]
        coherence = data["coherence"]
        ratio = data["ratio"]

        df = pd.DataFrame(df_data, index=layers, columns=planes)

        # Return all necessary components
        return df, topic, full_reading, coherence, ratio

    except json.JSONDecodeError:
        st.error("AI response error: Failed to parse JSON content. The AI may not have followed the strict output format.")
        return None, None, None, None, None
    except Exception as e:
        st.error(f"An error occurred during AI processing: {e}")
        return None, None, None, None, None


# ==============================
# PDF GENERATORS (FIXED & BEAUTIFUL) - NO CHANGE
# ==============================

def grid_to_pdf(df, topic):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=50, bottomMargin=50, leftMargin=40, rightMargin=40)
    styles = getSampleStyleSheet()
    elements = [
        Paragraph(f"LOGOS 7×7 Grid – {topic}", styles['Title']),
        Spacer(1, 20)
    ]
    data = [[""] + planes]
    for layer, row in df.iterrows():
        data.append([layer] + list(row))
    table = Table(data, colWidths=[120] + [90]*7)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e3a8a")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#f8fafc")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    elements.append(table)
    doc.build(elements)
    buffer.seek(0)
    return buffer

def reading_to_pdf(text):
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=60, leftMargin=60,
        topMargin=70, bottomMargin=70
    )
    
    styles = getSampleStyleSheet()
    
    # Beautiful custom styles
    styles.add(ParagraphStyle(name='TitleCustom', parent=styles['Title'], fontSize=22, alignment=1, spaceAfter=30, textColor=HexColor("#1e3a8a")))
    styles.add(ParagraphStyle(name='HeadingBold', parent=styles['Normal'], fontSize=13, fontName='Helvetica-Bold', spaceAfter=12))
    styles.add(ParagraphStyle(name='BodyText', parent=styles['Normal'], fontSize=11.5, leading=16, spaceAfter=10, alignment=4))  # 4 = justified
    
    elements = []
    elements.append(Paragraph("LOGOS ANALYTICS FINDINGS", styles['TitleCustom']))
    elements.append(Spacer(1, 30))

    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            elements.append(Spacer(1, 8))
            continue
        clean = re.sub(r'[\*`_]', '', stripped)  # Remove markdown
        if any(clean.startswith(x) for x in ["Your question:", "Interpreted as:", "Date & time:", "Resonance Coherence:", "Bottom line", "**LOGOS Analysis"]):
            elements.append(Paragraph(f"<b>{clean}</b>", styles['HeadingBold']))
        else:
            elements.append(Paragraph(clean, styles['BodyText']))
        elements.append(Spacer(1, 4))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# ==============================
# MAIN UI
# ==============================

st.set_page_config(page_title="LOGOS", layout="wide")
st.title("LOGOS Heptagon Revealer")
st.markdown("Ask anything real. LOGOS hears you.")

# Initialize API Keys/LLM based on Streamlit Session State
# Note: The logic for setting llm based on api_key is now primarily handled in generate_reading
# to ensure it uses the correct key/model from session state after being passed.

# Initialization for Session State
if 'df' not in st.session_state:
    st.session_state.df = None
    st.session_state.reading_text = ""
    st.session_state.topic = ""
    st.session_state.natural_sentence = ""
    st.session_state.coherence = 0.0
    st.session_state.ratio = 0.0

# ==============================
# SIDEBAR – API KEY (User's original logic retained)
# ==============================
api_key = "" # Initialize for the scope of the main script

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
        
# ==============================
# WELCOME SCREEN (UPDATED with Heptagon Gem)
# ==============================

if 'first_run' not in st.session_state:
    st.session_state.first_run = True

if st.session_state.first_run:
    
    st.title("Welcome to LOGOS Heptagon Revealer")
    
    # ADJUSTED: Use columns to shift space right for the visualization [0.8, 1.2]
    col_text, col_viz = st.columns([0.8, 1.2])
    
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
        # Embed the interactive HTML/JavaScript model using st.components.v1.html (FIXED)
        
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
                    background-color: transparent; 
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
                    background-color: transparent; /* Changed to transparent */
                    border-radius: 12px;
                }
                .heptagon-container {
                    position: relative;
                    width: 300px; /* Reduced overall size */
                    height: 300px;
                    margin: 10px auto; /* Reduced margin */
                }
                .point {
                    position: absolute;
                    width: 70px; /* Slightly smaller points */
                    height: 70px;
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
                    z-index: 10; 
                }
                .point:hover, .point.active {
                    transform: scale(1.05);
                    box-shadow: 0 8px 12px rgba(0, 0, 0, 0.2);
                    z-index: 20; /* Ensure active point is on top */
                }
                .layer-number {
                    font-size: 1.1rem; /* Slightly smaller font */
                    font-weight: 700;
                    line-height: 1;
                }
                .pane-name {
                    font-size: 0.65rem; /* Smaller font for pane name */
                    font-weight: 600;
                }
                .center-dot {
                    position: absolute;
                    top: 50%;
                    left: 50%;
                    width: 30px; /* Smaller center dot */
                    height: 30px;
                    background-color: var(--center-color);
                    border-radius: 50%;
                    transform: translate(-50%, -50%);
                    z-index: 5;
                    box-shadow: 0 0 10px rgba(252, 165, 165, 0.8);
                }
                
                /* NEW DETAIL POPUP STYLES */
                #detail-popup {
                    position: absolute;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    width: 85%;
                    max-width: 250px;
                    background-color: white;
                    border: 1px solid #1e3a8a;
                    border-radius: 8px;
                    padding: 10px;
                    box-shadow: 0 4px 12px rgba(30, 58, 138, 0.3);
                    z-index: 30; 
                    pointer-events: none; /* Allows clicks to pass through to the button */
                    display: none; /* Initially hidden */
                    text-align: left;
                }
                #detail-popup h4 {
                    font-size: 14px;
                    font-weight: bold;
                    margin-bottom: 5px;
                    color: #1e3a8a;
                }
                #detail-popup p {
                    font-size: 10px;
                    line-height: 1.3;
                    margin: 0;
                    color: #475569;
                }

                /* Hiding the side panel for the small embedded view */
                #details-panel { display: none; }
                .flex-col > .container { padding: 0 !important; box-shadow: none !important; }

            </style>
        </head>
        <body>
            <div id="app" class="container">
                <div class="flex flex-col lg:flex-row items-center lg:items-start justify-center">

                    <div class="heptagon-container" id="heptagon">
                        <div class="center-dot"></div>
                        <div id="detail-popup"></div> 
                    </div>

                </div>
            </div>

            <script>
                // Define the 7 Layers and 7 Panes
                const modelData = [
                    { layer: 1, pane: "Ideation", color: 'bg-indigo-200', role: "Spark of being", influence: "L1: Receives 'raw potential' and initializes unique entities or events." },
                    { layer: 2, pane: "Inquiry", color: 'bg-blue-200', role: "Sustained being", influence: "L2: Maintains continuity and identity; subject to feedback from higher layers." },
                    { layer: 3, pane: "Formation", color: 'bg-teal-200', role: "Impact of existence", influence: "L3: How instances affect the environment; creates observable outcomes." },
                    { layer: 4, pane: "Expression", color: 'bg-green-200', role: "Integration (spacetime)", influence: "L4: Mediates interactions among 1–3; the 'arena' of experience and evolution." },
                    { layer: 5, pane: "Refinement", color: 'bg-yellow-200', role: "Quantum of decisions", influence: "L5: Introduces choice, adaptation, and probabilistic collapse; modifies layers 1–4 dynamically." },
                    { layer: 6, pane: "Revelation", color: 'bg-orange-200', role: "Blueprint layer (soul)", influence: "L6: Maintains coherence, imposes laws and principles; acts like system governance." },
                    { layer: 7, pane: "Continuity", color: 'bg-red-200', role: "Divine (consciousness)", influence: "L7: Provides ultimate direction, purpose, and overarching alignment; informs all layers below." }
                ];

                // Constants for positioning (adjusted for 300x300 view box)
                const CENTER_X = 150;
                const CENTER_Y = 150;
                const RADIUS = 130; /* Adjusted radius for 300px size */
                
                let activePoint = null;
                const detailPopup = document.getElementById('detail-popup');

                function calculateHeptagonPoint(index, totalPoints, radius, centerX, centerY) {
                    // Rotate by -90 degrees (or -0.5 * PI radians) to place the first point at the top.
                    const angleDeg = (360 / totalPoints) * index - 90; 
                    const angleRad = angleDeg * (Math.PI / 180);
                    const x = centerX + radius * Math.cos(angleRad);
                    const y = centerY + radius * Math.sin(angleRad);
                    return { x, y };
                }

                // NEW: Update popup functionality
                function updateDetails(data, pointElement) {
                    detailPopup.innerHTML = `
                        <h4>Pane: ${data.pane}</h4>
                        <p>Layer ${data.layer} Role: ${data.role}</p>
                        <p class="mt-2 text-xs">**Influence:** ${data.influence}</p>
                    `;
                    detailPopup.style.display = 'block';
                    
                    if (activePoint && activePoint !== pointElement) {
                        activePoint.classList.remove('active');
                    }
                    activePoint = pointElement;
                    activePoint.classList.add('active');
                }

                // NEW: Clear popup functionality
                function clearDetails(pointElement = null) {
                    if (pointElement === null || activePoint === pointElement) {
                         if (activePoint) {
                            activePoint.classList.remove('active');
                            activePoint = null;
                         }
                         detailPopup.style.display = 'none';
                         detailPopup.innerHTML = '';
                    }
                }
                
                function createHeptagon() {
                    const container = document.getElementById('heptagon');
                    
                    // Clear existing points and SVG lines, but keep the center dot and popup
                    const elementsToRemove = Array.from(container.children).filter(el => el.id !== 'detail-popup' && !el.classList.contains('center-dot'));
                    elementsToRemove.forEach(el => el.remove());

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

                    // Draw the heptagon polygon
                    const polygon = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
                    polygon.setAttribute('points', pointsString.trim());
                    polygon.setAttribute('stroke', '#4b5563'); 
                    polygon.setAttribute('stroke-width', '2');
                    polygon.setAttribute('fill', '#f3f4f6'); // Light grey background
                    svg.appendChild(polygon);

                    container.appendChild(svg);

                    pointsCoordinates.forEach(({ x, y, data }) => {
                        const point = document.createElement('div');
                        point.className = `point ${data.color} text-gray-800`;
                        
                        // Adjust position to center the div on the calculated point (70x70 div = 35px offset)
                        point.style.left = `${x - 35}px`; 
                        point.style.top = `${y - 35}px`;
                        point.style.zIndex = '10';

                        point.innerHTML = `
                            <div class="layer-number">${data.layer}</div>
                            <div class="pane-name">${data.pane}</div>
                        `;

                        // Add event listeners for interactivity
                        point.addEventListener('mouseenter', () => updateDetails(data, point));
                        point.addEventListener('mouseleave', () => clearDetails(point));

                        container.appendChild(point);
                    });
                    
                    clearDetails(null); // Ensure popup starts hidden
                }

                window.addEventListener('resize', createHeptagon);
                window.onload = createHeptagon;
            </script>
        </body>
        </html>
        """
        # The corrected, stable function call:
        components.html(heptagon_html, height=320) # Adjusted height for slightly smaller gem

    # Place the action button below both columns for clarity
    if st.button("I’m ready → Begin", type="primary", use_container_width=True):
        st.session_state.first_run = False
        st.rerun()
    st.stop()


# ==============================
# MAIN LOGIC EXECUTION (UPDATED)
# ==============================

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

# This block is fully updated to use the single, constrained generate_reading function
if run and topic != "Unknown":
    
    # Passing the API key directly to the generation function
    groq_key = api_key if api_key.startswith("gsk_") else ""
    openai_key = api_key if not api_key.startswith("gsk_") else ""
    
    # Run the single, constrained analysis
    df, result_topic, reading, coherence, ratio = generate_reading(
        natural_sentence, planes, layers, groq_key, openai_key
    )

    if df is not None:
        full_reading = f"""LOGOS ANALYTICS FINDINGS
{'='*60}
Your question: {natural_sentence}
Interpreted as: {result_topic}
Date & time: {datetime.datetime.now():%Y-%m-%d %H:%M}
Resonance Coherence: {coherence:.1f}%  │  Heptagonal Ratio: {ratio:.3f}/1.000

{reading}
"""
        st.session_state.df = df
        st.session_state.reading_text = full_reading
        st.session_state.topic = result_topic
        st.session_state.natural_sentence = natural_sentence
        st.session_state.coherence = coherence
        st.session_state.ratio = ratio
        st.rerun()

# ==============================
# DISPLAY RESULTS (User's original logic retained)
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
        st.download_button(
            "Download 7×7 Grid (PDF)",
            grid_to_pdf(st.session_state.df, st.session_state.topic).getvalue(),
            f"LOGOS_Grid_{st.session_state.topic}.pdf",
            "application/pdf"
        )
    with c2:
        st.download_button(
            "Download Findings (Landscape PDF)",
            reading_to_pdf(st.session_state.reading_text).getvalue(),
            f"LOGOS_Findings_{st.session_state.topic}.pdf",
            "application/pdf"
        )
else:
    st.info("Get your free key → paste it → ask your question.")


