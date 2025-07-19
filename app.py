import streamlit as st
import pandas as pd
import plotly.express as px
import networkx as nx
import matplotlib.pyplot as plt
import requests
from fpdf import FPDF
import base64
import io
import tempfile 

st.set_page_config(page_title="NetCure.ai – Multi-Drug Predictions", layout="wide")

# Logo
st.image("/Users/nikitapatil/Downloads/netcure_mac_ready/app/net_logo.png", width=350)

st.title("NetCure.ai – Multi-Drug Target Predictions")
st.subheader("Drug Repositioning Predictions using Network-based AI")

# welcome text
st.write("""
Welcome to **NetCure.ai** — your AI-powered tool for discovering new uses for existing drugs 
using advanced network-based learning and real-time PubMed validation.
""")


# Sidebar explanations
with st.expander("About Us"):
    st.markdown("""
    **Who We Are**
    
    **NetCure.ai** is a pioneering AI-powered platform designed to transform how researchers, clinicians, and biotech innovators discover new therapeutic uses for existing drugs.

    Our mission is to make drug repositioning faster, smarter, and more evidence-based by combining powerful graph-based AI with real-time biomedical validation.

    ---

    **What We Do**
    
    NetCure.ai builds intelligent, multi-layered biomedical networks that connect drugs, diseases, genes, and proteins.  
    It predicts potential new drug–disease relationships that may otherwise remain hidden in isolated datasets.

    Each prediction is backed by real-time PubMed validation to ensure only the strongest, evidence-supported repositioning opportunities reach your screen.

    ---

    **How It Works**
    - **Data Integration:** Combines trusted biomedical sources such as DrugBank, PubMed, OMIM, and others.
    - **Graph-Based Learning:** Represents complex interactions in a network of up to 30,000 meaningful nodes.
    - **Smart Filtering:** Automatically removes low-confidence or noisy nodes to keep the model light and robust.
    - **Real-Time Validation:** Validates potential predictions against up-to-date PubMed entries, so you stay ahead with the latest findings.
    - **User-Friendly Interface:** Deployed with a clean, simple Streamlit web app for maximum accessibility.

    ---

    **Why NetCure.ai is Different**
    - **Network-First:** Unlike simple text mining tools, NetCure.ai understands and learns from the complex relationships between biomedical entities.
    - **Up-to-Date:** Your predictions are always supported by current research, thanks to PubMed API integration.
    - **Lightweight & Fast:** Node pruning keeps the graph model efficient and stable — even on personal devices.
    - **Easy to Use:** No advanced programming skills needed — just input, predict, and explore results visually.

    ---

    **Our Vision**
    NetCure.ai is built to empower scientists and healthcare professionals to find new treatments faster, reduce R&D costs, and bring promising repositioning candidates to light — all with a user-first, open-science spirit.

    **Together, let’s accelerate the discovery of new cures.**

    **Welcome to NetCure.ai platform!**
    """)
    
st.sidebar.title("How to Interpret Predictions")
with st.sidebar.expander("Confidence Score Explained"):
    st.write("The confidence score (0 to 1) shows how likely the drug–target link is real, "
        "based on the Node2Vec + Logistic Regression model. "
        "A higher score means stronger biological evidence.")
with st.sidebar.expander("Drug–Target Network Explained"):
    st.write("The network graph displays drugs (blue) and predicted targets (green). "
        "Green edges mean PubMed-validated, red means novel. "
        "Edge width shows prediction confidence.")
with st.sidebar.expander("Overall Prediction Summary"):
    st.write("These predictions include validated links (with PubMed IDs) and possible novel links. "
        "Use the tool to compare drugs, view PubMed-backed evidence, "
        "and export reports for your research.")

st.write("---")
st.write("Start exploring repositioning predictions now!")

# Load predictions
@st.cache_data
def load_predictions():
    df = pd.read_csv("data/processed/validated_predictions.tsv", sep="\t")
    df['Validation'] = df['PubMed_ID'].apply(lambda x: "Validated" if pd.notnull(x) and int(x) > 0 else "Not Validated")
    df['PubMed_Link'] = df['PubMed_ID'].apply(lambda x: f"https://pubmed.ncbi.nlm.nih.gov/{int(x)}/" if pd.notnull(x) and int(x) > 0 else "N/A")
    return df

df = load_predictions()
all_drugs = sorted(df['Drug'].unique())
selected_drugs = st.multiselect("Select Drug IDs", all_drugs, default=all_drugs[:1] if all_drugs else [])

if selected_drugs:
    sub_df = df[df['Drug'].isin(selected_drugs)]
    st.success(f"Showing {len(sub_df)} predictions for {len(selected_drugs)} drugs.")

    def make_pubmed_link(row):
        if row['Validation'] == "Validated":
            return f"[{int(row['PubMed_ID'])}]({row['PubMed_Link']})"
        else:
            return "N/A"

    sub_df['PubMed'] = sub_df.apply(make_pubmed_link, axis=1)

    st.dataframe(sub_df[['Drug', 'Drug_Name', 'Disease', 'Disease_Name', 'Confidence', 'Validation', 'Title', 'PubMed']])

    st.write("---")

    st.subheader("PubMed Abstracts")
    for _, row in sub_df.iterrows():
        if row['Validation'] == "Validated":
            if st.button(f"Show Full Abstract for PubMed {int(row['PubMed_ID'])}"):
                efetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
                params = {"db": "pubmed", "id": int(row['PubMed_ID']), "retmode": "xml"}
                response = requests.get(efetch_url, params=params)
                if response.status_code == 200:
                    try:
                        import xml.etree.ElementTree as ET
                        tree = ET.fromstring(response.content)
                        abstract = tree.find(".//AbstractText").text
                        st.info(f"**Abstract:** {abstract}")
                    except:
                        st.warning("Abstract not found in response.")
                else:
                    st.warning("Could not fetch abstract.")

    st.write("---")

    st.subheader("Combined Confidence Scores")
    fig = px.bar(sub_df, x='Disease_Name', y='Confidence', color='Drug_Name', barmode='group',
                 title="Prediction Confidence by Target")
    st.plotly_chart(fig, use_container_width=True)

    st.write("---")
    st.subheader("Advanced Drug–Target Network")

    G = nx.Graph()
    for _, row in sub_df.iterrows():
        drug_node = f"{row['Drug_Name']} ({row['Drug']})"
        disease_node = f"{row['Disease_Name']} ({row['Disease']})"
        G.add_node(drug_node, type='drug')
        G.add_node(disease_node, type='target')
        G.add_edge(drug_node, disease_node, weight=row['Confidence'], validation=row['Validation'])

    pos = nx.spring_layout(G)
    edge_colors = ['green' if G[u][v]['validation'] == 'Validated' else 'red' for u, v in G.edges()]
    edge_widths = [5 * G[u][v]['weight'] for u, v in G.edges()]
    node_colors = ['#1f78b4' if G.nodes[n]['type'] == 'drug' else '#33a02c' for n in G.nodes]
    fig2, ax = plt.subplots(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=800, font_size=8,
            edge_color=edge_colors, width=edge_widths, ax=ax)
    st.pyplot(fig2)

    st.write("---")
    st.subheader("Export All Predictions to PDF")
if st.button("Download PDF"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 12)
    title_text = "NetCure.ai - Ultimate Multi-Drug Prediction Report"
    safe_title = title_text.encode("latin-1", "replace").decode("latin-1")
    pdf.cell(0, 10, safe_title, ln=True)
    pdf.ln(5)

    # Table header
    pdf.set_font("Arial", "B", 10)
    col_width = pdf.w / 6.5  # Adjust column width to fit page
    headers = ["Drug", "Drug_Name", "Disease", "Disease_Name", "Confidence", "Validated"]
    for header in headers:
        safe_header = header.encode("latin-1", "replace").decode("latin-1")
        pdf.cell(col_width, 10, safe_header, border=1)
    pdf.ln()

    # Table rows
    pdf.set_font("Arial", "", 7)
    for _, row in sub_df.iterrows():
        row_data = [
            row['Drug'], row['Drug_Name'], row['Disease'],
            row['Disease_Name'], f"{row['Confidence']:.2f}", row['Validation']
        ]
        for cell in row_data:
            safe_cell = str(cell).encode("latin-1", "replace").decode("latin-1")
            pdf.cell(col_width, 10, safe_cell, border=1)
        pdf.ln()

    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    subtitle = "Combined Confidence Scores"
    safe_subtitle = subtitle.encode("latin-1", "replace").decode("latin-1")
    pdf.cell(0, 10, safe_subtitle, ln=True)
    pdf.ln(3)

    # Save bar chart with clear colors
    fig_bar = px.bar(
        sub_df, x='Disease_Name', y='Confidence', color='Drug_Name',
        barmode='group', title="Prediction Confidence by Target",
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    fig_bar.update_layout(title_text="Prediction Confidence by Target", plot_bgcolor="white")
    bar_buf = io.BytesIO()
    fig_bar.write_image(bar_buf, format='png')
    bar_buf.seek(0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(bar_buf.read())
        tmp.flush()
        pdf.image(tmp.name, w=170)

    pdf.ln(5)
    pdf.cell(0, 10, "Advanced Drug-Target Network", ln=True)
    pdf.ln(3)

    # Save network graph
    G = nx.Graph()
    for _, row in sub_df.iterrows():
        drug_node = f"{row['Drug_Name']} ({row['Drug']})"
        disease_node = f"{row['Disease_Name']} ({row['Disease']})"
        G.add_node(drug_node, type='drug')
        G.add_node(disease_node, type='target')
        G.add_edge(drug_node, disease_node, weight=row['Confidence'], validation=row['Validation'])

    pos = nx.spring_layout(G)
    edge_colors = ['green' if G[u][v]['validation'] == 'Validated' else 'red' for u, v in G.edges()]
    edge_widths = [5 * G[u][v]['weight'] for u, v in G.edges()]
    node_colors = ['#1f78b4' if G.nodes[n]['type'] == 'drug' else '#33a02c' for n in G.nodes]

    fig2, ax = plt.subplots(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color=node_colors,
            node_size=800, font_size=8,
            edge_color=edge_colors, width=edge_widths, ax=ax)
    net_buf = io.BytesIO()
    plt.savefig(net_buf, format='png', bbox_inches='tight')
    net_buf.seek(0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(net_buf.read())
        tmp.flush()
        pdf.image(tmp.name, w=170)

    pdf.ln(5)
    pdf.set_font("Arial", "", 11)
    summary_text = (
        "Summary: This report shows all validated and novel predictions in a clear table, "
        "confidence scores with color-coded bar chart, and an advanced drug–target network. "
        "Use PubMed IDs for evidence validation."
    )
    safe_summary = summary_text.encode("latin-1", "replace").decode("latin-1")
    pdf.multi_cell(0, 10, txt=safe_summary)

    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    buffer = io.BytesIO(pdf_bytes)
    b64 = base64.b64encode(buffer.read()).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="NetCure_Full_Report.pdf"> Click here to download the PDF report</a>'
    st.markdown(href, unsafe_allow_html=True)
   
st.write("---")

st.header("Contact Us")
with st.form("contact_form"):
    name = st.text_input("Your Name")  
    email = st.text_input("Your Email")
    message = st.text_area("Your Query Here")
    submitted = st.form_submit_button("Submit")

    if submitted:
        if not email:
            st.error("Please enter your email.")
        elif not message:
            st.error("Please enter your query.")
        else:
        
            st.success(f"Thanks, {name}! We received your query.")
            
st.write("---")
st.write("© 2025 NetCure.ai. All rights reserved. For research purpose only.")
