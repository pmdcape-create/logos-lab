import streamlit as st
import pandas as pd
import numpy as np
from sympy import Matrix, latex, symbols

# Hardcoded data from the workbook (unchanged)
laws = [
    {"ID": 1, "Name": "Law of Aligned Instantiation", "Description": "All instantiations (decisions, creations, actions) must align with Purpose", "Interaction": "Layer 6 (Blueprint), Layer 4 (Decision), Layer 1 (Purpose); Pane 1 ↔ Pane 6", "Check": lambda m: True},
    {"ID": 2, "Name": "Law of Quantum Instantiation", "Description": "Only the chosen decision is instantiated; unchosen options decohere", "Interaction": "Layer 4 ↔ Layer 6; Pane 4 → 5 → 6 → 7", "Check": lambda m: True},
    {"ID": 3, "Name": "Law of Systemic Feedback & Regulation", "Description": "The system self-regulates through indirect feedback loops", "Interaction": "Layer 5 ↔ Layer 6 ↔ Layer 7; Pane 5 → 6 → 7 → 1", "Check": lambda m: True},
    {"ID": 4, "Name": "Law of Consistency & Sustainability", "Description": "Must maintain internal coherence across time and planes", "Interaction": "Pane 3, 5, 7; Layer 3 ↔ Pane 3 etc.", "Check": lambda m: True},
    {"ID": 5, "Name": "Law of Guidance via Alignment with Higher Order", "Description": "Informed by top-down inputs (spiritual, archetypal, or systemic guidance)", "Interaction": "Pane 6, 5, 7; Layer 6 ↔ Pane 6 etc.", "Check": lambda m: True},
    {"ID": 6, "Name": "Law of Sustained Continuity", "Description": "Only aligned instantiations are carried forward in time", "Interaction": "Pane 7; Layer 7 ↔ Pane 7", "Check": lambda m: True},
    {"ID": 7, "Name": "Law of Telic Resolution", "Description": "A component that ceases to serve system purpose becomes dysfunctional", "Interaction": "Layer 1, 3; Pane 1 ↔ 3", "Check": lambda m: True},
    {"ID": 8, "Name": "Law of Iterative Collapse", "Description": "Decision collapses possibility into reality", "Interaction": "Layer 4 ↔ 5; Pane 4 ↔ Layer 5", "Check": lambda m: True},
    {"ID": 9, "Name": "Law of Bidirectional Blueprint Influence", "Description": "Brain and soul/conscience co-determine action and intention", "Interaction": "Layers 2–3 ↔ 6; Pane 2 ↔ 6", "Check": lambda m: True},
    {"ID": 10, "Name": "Law of Deferral-Resolution", "Description": "Deferred decisions reappear through system pressure", "Interaction": "Layer 4, 5, Pane 1; Pane 3 → 7 → 1", "Check": lambda m: True},
]

planes = ["Ideation", "Inquiry", "Formation", "Expression", "Refinement", "Revelation", "Continuity"]
layers = ["Instantiation", "Existence", "Effect / Impact", "Iteration", "Decision Quantum", "Blueprint / Soul", "Creator Layer"]

matrix_questions = [
    ["What initial intent or desire sets this idea into motion?", "What fundamental question arises at the moment of potential becoming?", "What seed structure or archetype is forming from this intent?", "How does the first form or presence of this idea appear?", "How is the original spark tested or filtered in early feedback?", "What does this moment of becoming reveal about the source or observer?", "What origin imprint will echo forward and shape recurring emergence?"],
    ["What meaning or purpose underlies this being’s presence?", "How does awareness explore identity and context here?", "How does the form experience or assert itself in relation to its environment?", "In what way does being express itself to others or the world?", "How does experience shape or challenge its original nature?", "What does this existence reveal about deeper truths or illusions?", "How is identity or presence preserved or forgotten across time and cycles?"],
    ["What intended outcome was embedded in this creation?", "What consequences arise, and what do they reflect about origin intent?", "How do effects begin shaping future states or structures?", "How is impact made visible, audible, or traceable in the world?", "How are results or consequences absorbed and recalibrated?", "What does the impact reveal about deeper laws (ethical, spiritual, systemic)?", "How does the echo of effect shape karmic, cultural, or energetic continuity?"],
    ["What cycles or repetitions are seeded by this?", "What patterns emerge that require questioning, renewal, or cessation?", "How is the form reassembled or carried forward into a new version?", "How does expression evolve across cycles or generations?", "What refinements are learned across lifetimes or iterations?", "What long-term insight arises through contrast, recursion, or pattern recognition?", "What keeps this pattern alive—or transforms it—through continuity of essence?"],
    ["Where is choice embedded in the fabric of this idea or being?", "What internal or external crossroads are being faced?", "How do decisions reshape the architecture of future reality?", "What actions externalize the choice made in this moment?", "How does consequence refine future decisions or rewire moral structure?", "What do the results of choice reveal about self, soul, or truth?", "How are decisions encoded as memory or trajectory across time and soul evolution?"],
    ["What archetypal or lawful pattern is seeded here?", "What does conscience or deep knowing reveal about the alignment of this path?", "How does the form harmonize (or clash) with inner blueprint or soul design?", "How does expression mirror the inner structure or soul’s design?", "What distortions are corrected as the soul returns to alignment?", "What divine pattern or origin signature is recognized through this process?", "How is the soul’s blueprint preserved, evolved, or restored over cycles?"],
    ["What infinite possibilities exist within the source of this inquiry?", "How is the Creator both asking and answering through this event?", "How is reality shaped as a divine thought-form given structure?", "In what way does the Creator express through this specific event or form?", "How does divine will or natural law refine the outcome through judgment?", "How does the Creator recognize itself through the unfolding of this moment?", "How is eternal continuity ensured through divine presence in all cycles?"],
]

st.title("LOGOS Model Logic Laboratory")

with st.sidebar:
    st.header("Guided Inquiry Template")
    for i, layer in enumerate(layers):
        st.subheader(layer)
        q = matrix_questions[i][0]
        st.write(q)
        st.text_input(f"Response for {layer}", key=f"inq_{i}")

st.header("User Input Submission")
user_input = st.text_area("Enter your topic, insight, question, or claim for analysis:")

if st.button("Run LOGOS Analysis"):
    if user_input:
        # Matrix population (unchanged)
        matrix_data = np.array([[f"Analysis for '{user_input}': {q}" for q in row] for row in matrix_questions])
        df = pd.DataFrame(matrix_data, index=layers, columns=planes)

        # Laws check (unchanged)
        violations = []
        for law in laws:
            if not law["Check"](matrix_data):
                violations.append(f"Violation of Law {law['ID']} ({law['Name']}): {law['Description']} - Interaction: {law['Interaction']}")

        st.subheader("Populated LOGOS 7x7 Matrix")
        st.dataframe(df)

        st.subheader("Systemic Laws Coherence Check")
        if violations:
            st.warning("Coherence issues detected:")
            for v in violations:
                st.write(v)
        else:
            st.success("All 10 laws satisfied; model is coherent.")

        # New: Mathematical Formalization Demo
        st.subheader("Mathematical Formalization Demo")
        st.write("Here, we formalize the heptagon as a 7x7 transition matrix \( M \), with state transitions \( \\vec{result} = M \\cdot \\vec{s} \).")
        
        # Define symbols for display
        T, r = symbols('T r')
        tension_eq = -T.diff(r)
        st.latex(f"\\nabla T = {latex(tension_eq)} \\quad (Tension Gradient for Gravity)")

        # Sample numerical M (proxy; could vectorize text via len(strings) or embeddings)
        M_num = np.eye(7)  # Identity for demo; replace with actual mappings
        s = np.array([len(' '.join(row)) / 1000.0 for row in matrix_data])  # Proxy vector from matrix content lengths
        result = M_num @ s

        sym_M = Matrix(M_num)
        sym_s = Matrix(s)
        sym_result = sym_M * sym_s

        st.latex(f"\\vec{{s}} = {latex(sym_s)}")
        st.latex(f"M = {latex(sym_M)}")
        st.latex(f"\\vec{{result}} = M \\cdot \\vec{{s}} = {latex(sym_result)}")

        st.write("Computed Transitioned State (numerical):")
        st.write(result)

        # Projection operator example
        st.latex("\\Pi(S_t) = \\bigoplus_{i=1}^7 P_i(S_t) \\quad (Heptagon Projection)")

        # Downloads (unchanged)
        csv_data = df.to_csv(index=True).encode('utf-8')
        st.download_button("Download Matrix as CSV", data=csv_data, file_name="logos_matrix.csv", mime="text/csv")

        excel_buffer = pd.ExcelWriter("logos_matrix.xlsx", engine='openpyxl')
        df.to_excel(excel_buffer, index=True)
        excel_buffer.close()
        with open("logos_matrix.xlsx", "rb") as f:
            st.download_button("Download Matrix as Excel", data=f, file_name="logos_matrix.xlsx")
    else:
        st.error("Please enter a topic to analyze.")

st.markdown("---")
st.caption("The LOGOS Model is a 7x7 heptagonal logic lab for analyzing existence from practical and theological views, with dynamic layers (1-5 physical, 6 spiritual, 7 divine) and laws ensuring coherence.")