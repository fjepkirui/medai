import json
import re
from typing import Any, List, Optional
import pandas as pd
import requests
import streamlit as st

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="DRG Predictor & Summarizer",
    page_icon="🧾",
    layout="wide",
)
st.title("🧾 DRG Predictor & Clinical Note Summarizer")

# -----------------------------
# Sidebar - configuration
# -----------------------------
with st.sidebar:
    st.header("Configuration ⚙️")
    MODEL_OPTIONS = [
        "Qwen2.5:7b-instruct",
        "Granite3.2:8b",
        "Llama3:latest",
        "Deepseek-r1:14b",
        ,
    ]
    model_name = st.selectbox("Default Model", MODEL_OPTIONS, index=0)

    num_predict_summary = st.number_input(
        "Max tokens for summary",
        min_value=32,
        max_value=2048,
        value=1500,
        step=16,
        help="Maximum tokens for generating clinical summaries",
    )

    num_predict_drg = st.number_input(
        "Max tokens for DRG code",
        min_value=1,
        max_value=100,
        value=50,
        step=1,
        help="Maximum tokens for DRG prediction (usually small, since output is just 3 digits)",
    )

    start_row = st.number_input(
        "Start row number", min_value=1, value=1, step=1
    )
    end_row = st.number_input("End row number", min_value=1, value=10, step=1)


# -----------------------------
# Ollama helper
# -----------------------------
def ollama_generate(
    prompt: str,
    model: str = "llama3:latest,"Qwen2.5:7b-instruct",
    options: dict | None = None,
    stop: List[str] | None = None,
    timeout: int = 180,
) -> str:
    """Call Ollama /api/generate and return concatenated text."""
    url = "http://localhost:11434/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": True}
    if options:
        payload["options"] = options
    if stop:
        payload["stop"] = stop

    chunks: List[str] = []
    try:
        with requests.post(url, json=payload, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line.decode("utf-8"))
                except Exception:
                    continue
                if "response" in data:
                    chunks.append(data["response"])
                elif "error" in data:
                    chunks.append(f"[OLLAMA ERROR] {data['error']}")
    except Exception as e:
        return f"[ERROR: {e}]"
    return "".join(chunks)


# -----------------------------
# Text utilities
# -----------------------------
def clean_text_blanks(s: Any) -> str:
    if not isinstance(s, str):
        return ""
    s = re.sub(r"[_\-]{3,}", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def format_drg_code(code: str) -> str:
    numbers = re.findall(r"\d+", str(code))
    if not numbers:
        return ""
    num_str = numbers[0][:3].zfill(3)
    return num_str


def extract_drg_code(text: str) -> str:
    if not isinstance(text, str):
        return ""
    match = re.search(r"DRG-CODE:\s*(\d{3})", text, re.IGNORECASE)
    if match:
        return match.group(1)
    nums = re.findall(r"\b\d{3}\b", text)
    return nums[0] if nums else ""


# -----------------------------
# Prompt builders
# -----------------------------
DEFAULT_DRG_PROMPT = """
You are an expert medical coder specializing in MS-DRG assignment.

TASK: Analyze the clinical note below and output ONLY the most appropriate 3-digit MS-DRG code.

CRITICAL INSTRUCTIONS:
- Output MUST be in this exact format: DRG-CODE: XXX
- Replace XXX with the 3-digit DRG code
- DO NOT include any other text, explanations, or punctuation
- DO NOT write sentences or paragraphs
- DO NOT include ICD codes or other information

CLINICAL NOTE:
{clinical_text_here}

DRG-CODE:
"""

def build_summary_prompt(ehr_text: str) -> str:
    return f"""
Please provide a concise clinical summary of the following patient encounter. 
Focus on key medical issues, treatments, procedures, and outcomes.

<patient_record>
{ehr_text}
</patient_record>

Clinical Summary:
""".strip()

def build_drg_prompt(clinical_text: str) -> str:
    return DEFAULT_DRG_PROMPT.replace("{clinical_text_here}", clinical_text)


# -----------------------------
# File helpers
# -----------------------------
def process_table_file(uploaded_file) -> pd.DataFrame:
    try:
        name = uploaded_file.name.lower()
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Upload CSV or Excel (.xlsx/.xls).")
            return pd.DataFrame()
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return pd.DataFrame()


def detect_text_column(df: pd.DataFrame) -> Optional[str]:
    df_cols_lower = {col.lower().strip(): col for col in df.columns}
    for name in ["text", "clinical_text", "note", "clinical_note", "description", "content"]:
        if name in df_cols_lower:
            return df_cols_lower[name]
    return None


def detect_label_column(df: pd.DataFrame) -> Optional[str]:
    df_cols_lower = {col.lower().strip(): col for col in df.columns}
    for name in ["label", "drg_code", "drg", "ms_drg", "ms-drg", "code", "target", "class"]:
        if name in df_cols_lower:
            return df_cols_lower[name]
    return None


# -----------------------------
# Batch Processing (Multi-model)
# -----------------------------
def batch_predict_multi_model(
    df: pd.DataFrame,
    text_col: str,
    label_col: Optional[str],
    start_idx: int,
    end_idx: int,
    models: List[str],
    num_predict: int,
) -> pd.DataFrame:
    """Process multiple rows across multiple models."""
    results = []
    total = end_idx - start_idx
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, idx in enumerate(range(start_idx, end_idx)):
        progress = (i + 1) / total
        progress_bar.progress(progress)
        status_text.text(f"Processing row {idx + 1} of {end_idx}...")

        clinical_text = clean_text_blanks(df.iloc[idx][text_col])
        actual_drg = ""
        if label_col:
            actual_drg = format_drg_code(str(df.iloc[idx][label_col]))

        row_result = {"Row": idx + 1, "Actual_DRG": actual_drg if actual_drg else "N/A"}

        for model in models:
            prompt = build_drg_prompt(clinical_text)
            drg_raw = ollama_generate(
                prompt,
                model=model,
                options={"temperature": 0.01, "num_predict": num_predict},
                stop=["\n", ".", "DRG-CODE:", "DRG-CODE"],
            )
            predicted_drg = extract_drg_code(drg_raw)
            row_result[f"{model}_Predicted"] = predicted_drg if predicted_drg else "FAILED"

        results.append(row_result)

    progress_bar.empty()
    status_text.empty()
    return pd.DataFrame(results)


# -----------------------------
# Tabs (flow)
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Paste Text", "Upload File", "Summarize Notes", "Predict DRG", "Batch Process"]
)

if "clinical_text" not in st.session_state:
    st.session_state["clinical_text"] = ""

# --- Tab 1: Paste Text ---
with tab1:
    st.header("Paste Clinical Text")
    ehr_text_raw = st.text_area("Paste encounter note:", height=300)
    ehr_text = clean_text_blanks(ehr_text_raw)
    if ehr_text:
        st.success(f"Text loaded ({len(ehr_text)} characters)")
        st.session_state["clinical_text"] = ehr_text

# --- Tab 2: Upload File ---
with tab2:
    st.header("Upload CSV or Excel File")
    uploaded_file = st.file_uploader("Choose file", type=["csv", "xlsx", "xls"])
    if uploaded_file is not None:
        df = process_table_file(uploaded_file)
        if df.empty:
            st.warning("Uploaded file could not be read or is empty.")
        else:
            st.success(f"File loaded: {len(df)} rows, {len(df.columns)} columns")
            text_col = detect_text_column(df)
            label_col = detect_label_column(df)
            all_cols = list(df.columns)

            if not text_col:
                text_col = st.selectbox("Select text column", all_cols)
            if not label_col:
                label_col = st.selectbox("Select label column (optional)", [None] + all_cols)

            st.session_state["uploaded_df"] = df
            st.session_state["text_col"] = text_col
            st.session_state["label_col"] = label_col

# --- Tab 3: Summarize Notes ---
with tab3:
    st.header("Summarize Clinical Note")
    clinical_text = st.session_state.get("clinical_text", "")
    if not clinical_text:
        st.warning("Please provide clinical text first.")
    else:
        if st.button("Generate Summary"):
            with st.spinner("Generating summary..."):
                prompt = build_summary_prompt(clinical_text)
                summary = ollama_generate(
                    prompt,
                    model=model_name,
                    options={"temperature": 0.1, "num_predict": num_predict_summary},
                )
            summary = clean_text_blanks(summary)
            st.text_area("Clinical Summary", value=summary, height=200)
            st.session_state["generated_summary"] = summary

# --- Tab 4: Predict DRG ---
with tab4:
    st.header("Predict MS-DRG Code")
    clinical_text = st.session_state.get("clinical_text", "")
    if not clinical_text:
        st.warning("Please provide clinical text first.")
    else:
        default_prompt = build_drg_prompt(clinical_text)
        prompt_text = st.text_area("DRG Prediction Prompt", value=default_prompt, height=350)
        if st.button("Predict DRG"):
            with st.spinner("Predicting..."):
                drg_raw = ollama_generate(
                    prompt_text,
                    model=model_name,
                    options={"temperature": 0.01, "num_predict": num_predict_drg},
                    stop=["\n", ".", "DRG-CODE:", "DRG-CODE"],
                )
            drg_code = extract_drg_code(drg_raw)
            if drg_code:
                st.success(f"MS-DRG: {drg_code}")
            else:
                st.error("Could not extract a valid 3-digit DRG code.")

# --- Tab 5: Batch Process (Multi-model) ---
with tab5:
    st.header(" Batch Process DRG Predictions (Multiple Models)")
    if "uploaded_df" not in st.session_state:
        st.warning("Please upload a file in Tab 2 first.")
    else:
        df = st.session_state["uploaded_df"]
        text_col = st.session_state.get("text_col")
        label_col = st.session_state.get("label_col")

        if not text_col:
            st.error("Text column not identified. Please configure in Tab 2.")
        else:
            st.info(f"**Rows:** {len(df)}")
            st.info(f"**Text column:** {text_col}")
            st.info(f"**Label column:** {label_col if label_col else 'None'}")

            model_options = ["tinyllama", "mistral", "llama3", "qwen", "codellama", "deepseek"]
            selected_models = st.multiselect(
                "Select models to run concurrently",
                model_options,
                default=["tinyllama", "mistral"],
            )

            sr = int(max(1, start_row))
            er = int(min(len(df), end_row))
            if sr > er:
                sr = er

            st.markdown(f"### Will process rows {sr} to {er} ({er - sr + 1} rows)")

            col1, col2 = st.columns([1, 3])
            with col1:
                run_batch = st.button(" Start Batch Processing", type="primary")
            with col2:
                clear_results = st.button("Clear Results")

            if clear_results and "batch_results" in st.session_state:
                del st.session_state["batch_results"]
                st.success("Results cleared.")

            if run_batch:
                with st.spinner("Running multi-model batch..."):
                    results_df = batch_predict_multi_model(
                        df=df,
                        text_col=text_col,
                        label_col=label_col,
                        start_idx=sr - 1,
                        end_idx=er,
                        models=selected_models,
                        num_predict=num_predict_drg,
                    )
                    st.session_state["batch_results"] = results_df
                    st.success("✅ Batch processing complete!")

            if "batch_results" in st.session_state:
                results_df = st.session_state["batch_results"]
                st.dataframe(results_df, use_container_width=True)
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "⬇️ Download Results as CSV",
                    data=csv,
                    file_name="multi_model_drg_predictions.csv",
                    mime="text/csv",
                )
