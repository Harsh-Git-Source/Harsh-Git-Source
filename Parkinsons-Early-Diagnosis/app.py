import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Parkinson's Early Screening",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1100px;}
.hero {padding: 1.5rem; border-radius: 16px; background: #f4f7fb; margin-bottom: 1.5rem;}
.notice {padding: 1rem; border-radius: 12px; background: #fff8e6; border: 1px solid #f0d98c;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>🧠 Parkinson's Early Screening</h1>
<p>Questionnaire-based research prototype using ML and FLAN-T5.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="notice">
<b>Research prototype:</b> This application is for education and demonstration.
It does not provide a medical diagnosis. If you have health concerns, consult a qualified clinician.
</div>
""", unsafe_allow_html=True)

st.divider()

st.subheader("Questionnaire")
st.caption("The production interface should be populated with the project's authorized questionnaire question bank.")

questions = [
    ("q1", "Have you experienced noticeable changes in your sense of smell?"),
    ("q2", "Have you experienced tremor or shaking at rest?"),
    ("q3", "Have you noticed changes in walking or balance?"),
    ("q4", "Have you experienced stiffness or rigidity?"),
    ("q5", "Have you noticed changes in handwriting?"),
    ("q6", "Have you experienced changes in voice volume or clarity?"),
]

responses = {}
cols = st.columns(2)
for i, (key, q) in enumerate(questions):
    with cols[i % 2]:
        responses[key] = st.radio(
            q,
            ["Yes", "No", "Prefer not to answer"],
            key=key,
            horizontal=True
        )

if st.button("Analyze screening responses", type="primary", use_container_width=True):
    st.subheader("Screening result")

    # Deliberately do not invent a medical prediction when a trained checkpoint
    # is not present. This keeps demo mode safe and honest.
    checkpoint = Path("models/flan_t5_classifier")
    if not checkpoint.exists():
        st.info(
            "Demo mode: no trained clinical model checkpoint is included in this repository. "
            "Train a model on authorized PPMI data before enabling model-based inference."
        )
        st.write("Captured responses:")
        st.json(responses)
    else:
        st.warning("Checkpoint detected. Connect the trained inference wrapper here.")

st.divider()
st.caption("Built as a research/portfolio implementation of the supplied Parkinson's Disease ML + LLM project.")
