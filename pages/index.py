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
    layout="centered",
)
st.title("🧾 DRG Predictor & Clinical Note Summarizer")

# -----------------------------
# Sidebar - configuration
# -----------------------------
with st.sidebar:
    st.header("Configuration ⚙️")
    MODEL_OPTIONS = [
        "tinyllama",
        "deepseek",
        "llama3",
        "mistral",
        "qwen",
        "codellama",
    ]
    model_name = st.selectbox("Model", MODEL_OPTIONS, index=0)

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
        value=50,  # Reduced since we only need 3 digits
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
    model: str = "tinyllama",
    options: dict | None = None,
    stop: List[str] | None = None,
    timeout: int = 180,
) -> str:
    """
    Call Ollama /api/generate (streaming) and return concatenated text.
    """
    url = "http://localhost:11434/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": True}
    if options:
        payload["options"] = options
    if stop:
        payload["stop"] = stop

    chunks: List[str] = []
    try:
        with requests.post(
            url, json=payload, stream=True, timeout=timeout
        ) as r:
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


DRG_REGEX = re.compile(r"^\d{3}$")


def format_drg_code(code: str) -> str:
    if not code or not isinstance(code, str):
        return ""
    numbers = re.findall(r"\d+", code)
    if not numbers:
        return ""
    num_str = numbers[0]
    if len(num_str) > 3:
        num_str = num_str[:3]
    elif len(num_str) < 3:
        num_str = num_str.zfill(3)
    return num_str


def extract_drg_code(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    # Look for DRG-CODE: pattern first
    drg_pattern = re.search(r"DRG-CODE:\s*(\d{3})", text, re.IGNORECASE)
    if drg_pattern:
        return drg_pattern.group(1)

    # Then look for any 3-digit number
    nums = re.findall(r"\b\d{3}\b", text)
    if nums:
        return nums[0]

    return ""


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
- If uncertain, choose the most likely DRG code based on the clinical presentation

CLINICAL NOTE:
{clinical_text_here}

DRG-CODE:"""


def build_summary_prompt(ehr_text: str) -> str:
    return f"""
Please provide a concise clinical summary of the following patient encounter. 
Focus on key medical issues, treatments, procedures, and outcomes. Keep it brief and clinical.

<patient_record>
{ehr_text}
</patient_record>

Clinical Summary:
""".strip()


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
            st.error(
                "Unsupported file format. Upload CSV or Excel (.xlsx/.xls)."
            )
            return pd.DataFrame()
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return pd.DataFrame()


def detect_text_column(df: pd.DataFrame) -> Optional[str]:
    df_cols_lower = {col.lower().strip(): col for col in df.columns}
    for possible_name in [
        "text",
        "clinical_text",
        "note",
        "clinical_note",
        "description",
        "content",
    ]:
        if possible_name in df_cols_lower:
            return df_cols_lower[possible_name]
    return None


def detect_label_column(df: pd.DataFrame) -> Optional[str]:
    df_cols_lower = {col.lower().strip(): col for col in df.columns}
    for possible_name in [
        "label",
        "drg_code",
        "drg",
        "ms_drg",
        "ms-drg",
        "code",
        "target",
        "class",
    ]:
        if possible_name in df_cols_lower:
            return df_cols_lower[possible_name]
    return None


# -----------------------------
# Tabs (flow)
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["Paste Text", "Upload File", "Summarize Notes", "Predict DRG"]
)

# Initialize session state for clinical text
if "clinical_text" not in st.session_state:
    st.session_state["clinical_text"] = ""

# --- Tab 1: Paste Text ---
with tab1:
    st.header("Paste Clinical Text")
    ehr_text_raw = st.text_area(
        "Paste encounter note:",
        height=300,
        placeholder="Paste clinical note text here...",
        key="text_input",
    )
    ehr_text = clean_text_blanks(ehr_text_raw)
    if ehr_text:
        st.success(f"Text loaded ({len(ehr_text)} characters)")
        with st.expander("View cleaned text"):
            st.text(ehr_text)
        # Store the clinical text for DRG prediction
        st.session_state["clinical_text"] = ehr_text

# --- Tab 2: Upload File ---
with tab2:
    st.header("Upload CSV or Excel File")
    st.markdown(
        """
    **Requirements**
    - Must contain a column with clinical text.
    - Optionally may contain a label column with true DRG codes.
    """
    )
    uploaded_file = st.file_uploader(
        "Choose CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        key="file_upload",
    )

    if uploaded_file is not None:
        df = process_table_file(uploaded_file)
        if df.empty:
            st.warning("Uploaded file could not be read or is empty.")
        else:
            st.success(
                f"File loaded: {len(df)} rows, {len(df.columns)} columns"
            )
            text_col = detect_text_column(df)
            label_col = detect_label_column(df)
            all_cols = list(df.columns)

            if not text_col:
                text_col = st.selectbox(
                    "Select text column",
                    options=all_cols,
                    key="manual_text_col",
                )
            else:
                text_col = st.selectbox(
                    "Text column",
                    options=all_cols,
                    index=all_cols.index(text_col),
                    key="text_col_select",
                )

            label_col = st.selectbox(
                "Label column (optional)",
                options=[None] + all_cols,
                index=0 if not label_col else (all_cols.index(label_col) + 1),
                key="label_col_select",
            )

            nrows = len(df)
            sr = int(max(1, start_row))
            er = int(min(nrows, end_row)) if nrows >= 1 else 1
            if sr > er:
                sr = er

            st.markdown(f"**Previewing rows {sr} to {er}**")
            preview_df = df.iloc[sr - 1 : er].copy()
            cols_display = [c for c in [text_col, label_col] if c] + [
                c
                for c in df.columns
                if c not in ([text_col, label_col] if label_col else [text_col])
            ]
            preview_df = preview_df[cols_display]
            st.dataframe(preview_df.reset_index(drop=True))

            preview_indices = list(range(sr - 1, er))

            def format_row_choice(i):
                txt = str(df.iloc[i][text_col])
                snippet = txt[:120].replace("\n", " ")
                return (
                    f"Row {i+1}: {snippet}..."
                    if len(snippet) < len(txt)
                    else f"Row {i+1}: {snippet}"
                )

            selected_choice = st.selectbox(
                "Select a row to analyze",
                options=preview_indices,
                format_func=format_row_choice,
                key="row_select",
            )
            selected_text = clean_text_blanks(
                df.iloc[selected_choice][text_col]
            )
            # Store the clinical text for DRG prediction
            st.session_state["clinical_text"] = selected_text
            st.session_state["uploaded_df"] = df
            st.session_state["text_col"] = text_col
            st.session_state["label_col"] = label_col
            with st.expander("View extracted text"):
                st.text(selected_text)
            if label_col:
                actual_label = df.iloc[selected_choice][label_col]
                st.info(f"Actual label (if present): {actual_label}")

# --- Tab 3: Summarize Notes ---
with tab3:
    st.header("Summarize Clinical Note")
    clinical_text = st.session_state.get("clinical_text", "")
    if not clinical_text:
        st.warning(
            "Please paste text in Tab 1 or upload/select a row in Tab 2 first."
        )
    else:
        if st.button("Generate Summary", key="gen_summary"):
            with st.spinner("Calling model to generate summary..."):
                prompt = build_summary_prompt(clinical_text)
                summary = ollama_generate(
                    prompt,
                    model=model_name,
                    options={
                        "temperature": 0.1,
                        "num_predict": num_predict_summary,
                    },
                )
            summary = clean_text_blanks(summary)
            st.success("Summary generated")
            st.text_area(
                "Clinical Summary (editable)",
                value=summary,
                height=200,
                key="summary_area",
            )
            st.session_state["generated_summary"] = summary
        else:
            prev = st.session_state.get("generated_summary", "")
            if prev:
                st.text_area(
                    "Clinical Summary (editable)",
                    value=prev,
                    height=200,
                    key="summary_area_existing",
                )
            else:
                st.info(
                    "No summary yet. Click 'Generate Summary' to create one."
                )

# --- Tab 4: Predict DRG ---
with tab4:
    st.header("Predict MS-DRG Code")
    clinical_text = st.session_state.get("clinical_text", "")
    if not clinical_text:
        st.warning("Please provide clinical text in Tab 1 or Tab 2 first.")
    else:
        default_prompt_filled = DEFAULT_DRG_PROMPT.replace(
            "{clinical_text_here}", clinical_text
        )
        prompt_text = st.text_area(
            "DRG Prediction Prompt (editable)",
            value=default_prompt_filled,
            height=350,
            key="drg_prompt_area",
        )

        if st.button("Predict DRG", key="predict_btn"):
            with st.spinner("Calling model for DRG prediction..."):
                # Use lower temperature and add stop words to prevent extra text
                drg_raw = ollama_generate(
                    prompt_text,
                    model=model_name,
                    options={
                        "temperature": 0.01,  # Very low temperature for deterministic output
                        "num_predict": num_predict_drg,
                    },
                    stop=[
                        "\n",
                        ".",
                        "DRG-CODE:",
                        "DRG-CODE",
                    ],  # Stop at newlines or if it repeats the pattern
                )

            st.write(
                "🔍 Raw model output:", repr(drg_raw)
            )  # Use repr to see exact output

            # Extract DRG code
            drg_code = extract_drg_code(drg_raw)

            if drg_code:
                st.success("DRG code extracted successfully!")
                st.markdown(f"# **MS-DRG: {drg_code}**")

                # Show what was extracted
                with st.expander("Extraction details"):
                    st.write(f"Raw output: `{drg_raw}`")
                    st.write(f"Extracted code: `{drg_code}`")
            else:
                st.error("Could not extract a valid 3-digit DRG code.")
                with st.expander("Debug information"):
                    st.write("Raw output:", repr(drg_raw))
                    st.write("Attempted extraction from:", drg_raw)

                # Manual override
                st.subheader("Manual Entry")
                manual = st.text_input(
                    "Enter DRG code manually (3 digits)",
                    max_chars=3,
                    key="manual_drg",
                    help="Enter a 3-digit DRG code if automatic extraction failed",
                )
                if manual:
                    manual_clean = clean_text_blanks(manual)
                    if re.fullmatch(r"\d{3}", manual_clean):
                        st.success(f"Manual MS-DRG set to: {manual_clean}")
                        st.markdown(f"# **MS-DRG: {manual_clean}**")
                    else:
                        st.error("Please enter exactly 3 digits")

        if st.button("Reset DRG Prompt", key="reset_prompt"):
            default_reset = DEFAULT_DRG_PROMPT.replace(
                "{clinical_text_here}", clinical_text
            )
            st.session_state["drg_prompt_area"] = default_reset
            st.success("Prompt reset to default")

# -----------------------------
# End of app
# -----------------------------
st.markdown("---")
st.caption(
    "This app uses a local Ollama instance. Use only de-identified notes."
)