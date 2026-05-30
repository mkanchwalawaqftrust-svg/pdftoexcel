import streamlit as st
import pdfplumber
import pandas as pd
import re
import os
import tempfile
import io
import time
import datetime
from pypdf import PdfReader

# ==========================================
# 1. CORE ENGINE UTILITIES
# ==========================================

LINE_SCALE_DEFAULT = 40

def get_total_pages(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        return len(reader.pages)
    except Exception as e:
        st.error(f"Error reading PDF page count: {e}")
        return 0

def normalize_header_row(series):
    normalized = []
    for val in series:
        s = str(val).strip().lower()
        s = re.sub(r'\s+', ' ', s)
        if s in ['', 'nan', 'none', '<na>']:
            s = ''
        normalized.append(s)
    return normalized

def are_headers_matching(row1, row2):
    if len(row1) != len(row2):
        return False
    return normalize_header_row(row1) == normalize_header_row(row2)

def is_likely_header(row):
    if len(row) == 0:
        return False
    
    text_count = 0
    numeric_count = 0
    header_keywords = {'date', 'desc', 'amount', 'total', 'id', 'name', 'qty', 'rate', 'price', 'balance', 'account', 'mauze', 'jamaat', 'jamiat'}
    
    for val in row:
        val_str = str(val).strip().lower()
        if val_str in ['', 'nan', 'none']:
            continue
        if re.search(r'^\d+[\d.,]*%?$', val_str) or re.search(r'^[\$\€\£\₹]?\s?\d+', val_str):
            numeric_count += 1
        else:
            text_count += 1
            
    if numeric_count > 0:
        return False
        
    row_words = set(re.sub(r'\s+', ' ', str(val)).strip().lower() for val in row)
    if row_words.intersection(header_keywords):
        return True
        
    return text_count > 0

def align_table_header(df):
    if df.empty:
        return df
        
    for i in range(min(5, len(df))):
        row = df.iloc[i]
        if is_likely_header(row):
            aligned_df = df.iloc[i:].copy().reset_index(drop=True)
            return aligned_df
            
    return df

def detect_column_indices(df, header_row):
    normalized_headers = normalize_header_row(header_row)
    
    desc_keywords = ['desc', 'detail', 'particular', 'item', 'transaction', 'name', 'mauze']
    anchor_keywords = ['amount', 'total', 'balance', 'id', 'ref', 'debit', 'credit', 'price', 'val', 'cost', 'no', 'num', 'date', 'inst']
    
    desc_idx = 0  
    anchor_idx = -1
    
    for i, header in enumerate(normalized_headers):
        if any(keyword in header for keyword in desc_keywords):
            desc_idx = i
            break
            
    for i, header in enumerate(normalized_headers):
        if i != desc_idx and any(keyword in header for keyword in anchor_keywords):
            anchor_idx = i
            break
            
    if anchor_idx == -1:
        best_col = -1
        max_density = -1
        data_rows = df.iloc[1:] if len(df) > 1 else df
        
        for col in data_rows.columns:
            if col == desc_idx:
                continue
            fill_count = sum(1 for val in data_rows[col] if str(val).strip() not in ['', 'nan', 'None'])
            density = fill_count / len(data_rows) if len(data_rows) > 0 else 0
            if density > max_density:
                max_density = density
                best_col = col
                
        anchor_idx = best_col if best_col != -1 else (len(header_row) - 1 if len(header_row) > 0 else 0)
            
    return desc_idx, anchor_idx

# ==========================================
# 2. PROCESSING PIPELINE STAGES
# ==========================================

def clean_ghost_rows_safe(df, desc_col_idx, anchor_col_idx):
    if len(df) <= 1: 
        return df
        
    header_df = df.iloc[[0]]
    data_df = df.iloc[1:].copy().reset_index(drop=True)
    
    col_count = len(data_df.columns)
    rows_to_drop = []
    artifact_patterns = re.compile(r'^(page \d|run date|printed on|report id|confidential|user:)', re.IGNORECASE)
    
    for i in range(len(data_df)):
        row = data_df.iloc[i]
            
        fill_count = sum(1 for val in row if str(val).strip() not in ['', 'nan', 'None'])
        if fill_count == 0:
            rows_to_drop.append(i)
            continue
            
        if i < 3 or i >= len(data_df) - 3:
            anchor_val = str(row.iloc[anchor_col_idx]).strip() if anchor_col_idx < col_count else ""
            desc_val = str(row.iloc[desc_col_idx]).strip() if desc_col_idx < col_count else ""
            
            if anchor_val not in ['', 'nan', 'None']:
                continue
                
            if anchor_val in ['', 'nan', 'None'] and desc_val not in ['', 'nan', 'None']:
                if artifact_patterns.search(desc_val):
                    rows_to_drop.append(i)
                continue
            
            density = fill_count / col_count
            if density < 0.20:
                rows_to_drop.append(i)

    cleaned_data = data_df.drop(index=rows_to_drop).reset_index(drop=True)
    return pd.concat([header_df, cleaned_data], ignore_index=True)

def zip_wrapped_text_optimized(df, desc_col_idx, anchor_col_idx):
    if len(df) <= 1 or len(df.columns) <= max(desc_col_idx, anchor_col_idx):
        return df
        
    header_df = df.iloc[[0]]
    data_df = df.iloc[1:].copy().reset_index(drop=True)
    
    rows = data_df.values.tolist()
    columns = data_df.columns
    optimized_rows = []
    parent_idx = -1 
    consecutive_merges = 0  
    
    for row in rows:
        anchor_val = str(row[anchor_col_idx]).strip()
        desc_val = str(row[desc_col_idx]).strip()
        
        is_anchor_empty = anchor_val in ['', 'nan', 'None']
        is_desc_empty = desc_val in ['', 'nan', 'None']
        
        if is_anchor_empty and not is_desc_empty and parent_idx != -1 and consecutive_merges < 2:
            parent_row = optimized_rows[parent_idx]
            parent_desc = str(parent_row[desc_col_idx]).strip()
            parent_row[desc_col_idx] = f"{parent_desc} {desc_val}".strip()
            consecutive_merges += 1
        else:
            optimized_rows.append(row)
            consecutive_merges = 0  
            if not is_anchor_empty:
                parent_idx = len(optimized_rows) - 1
                
    zipped_data = pd.DataFrame(optimized_rows, columns=columns)
    return pd.concat([header_df, zipped_data], ignore_index=True)

def remove_duplicate_headers(df, header_row):
    if len(df) <= 1:
        return df
        
    header_df = df.iloc[[0]]
    data_df = df.iloc[1:].copy().reset_index(drop=True)
    
    normalized_target = normalize_header_row(header_row)
    rows_to_drop = []
    
    for i in range(len(data_df)):
        current_row_normalized = normalize_header_row(data_df.iloc[i])
        if current_row_normalized == normalized_target:
            rows_to_drop.append(i)
            
    cleaned_data = data_df.drop(index=rows_to_drop).reset_index(drop=True)
    return pd.concat([header_df, cleaned_data], ignore_index=True)

def parse_date_preserving_hyphen(val):
    """
    Parses date strings to datetime.date objects for Excel,
    but protects hyphens from being converted [1].
    """
    val_str = str(val).strip()
    if val_str in ['-', '—', '–', '.-']:
        return val_str  # Preserve hyphen as raw string
    if val_str in ['', 'nan', 'None']:
        return None
        
    try:
        dt = pd.to_datetime(val_str, errors='raise')
        return dt.date()  # Returns date-only object
    except Exception:
        return val_str  # Fallback to string if unparseable

def clean_raw_strings_and_type_dates(df):
    """
    Keeps string structures completely untouched, but dynamically identifies 
    date columns to apply native datetime.date objects for proper Excel date filters [1].
    """
    if len(df) <= 1:
        return df
        
    header_df = df.iloc[[0]]
    data_df = df.iloc[1:].copy().reset_index(drop=True)
    
    for col in data_df.columns:
        sample_series = data_df[col].dropna().astype(str).str.strip()
        sample_series = sample_series[~sample_series.isin(['', 'nan', 'None'])]
        
        if sample_series.empty:
            data_df[col] = None
            continue
            
        # Detect if this column contains dates
        date_like_count = sample_series.str.contains(r'\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}').sum()
        if (date_like_count / len(sample_series)) >= 0.75:
            # Apply hybrid date-preservation parser [1]
            data_df[col] = data_df[col].apply(parse_date_preserving_hyphen)
            continue
            
        # For non-date columns, preserve exactly as string, replacing nan with None
        data_df[col] = data_df[col].apply(
            lambda x: None if str(x).strip() in ['nan', 'None'] else str(x).strip()
        )
        
    return pd.concat([header_df, data_df], ignore_index=True)

# ==========================================
# 4. STREAMLIT FRONT-END UI & LAYOUT
# ==========================================

st.set_page_config(
    page_title="PDF to Excel",
    page_icon="📊",
    layout="centered"
)

st.title("📊 PDF to Excel")
st.write("Extract bordered PDF tables with exact-copy string matching and automated header alignment.")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(uploaded_file.read())
        temp_pdf_path = temp_pdf.name
        
    total_pages_pdf = get_total_pages(temp_pdf_path)
    
    st.info(f"📄 Detected {total_pages_pdf} pages in **{uploaded_file.name}**")
    
    col1, col2 = st.columns(2)
    with col1:
        start_page = st.number_input("Start Page", min_value=1, max_value=total_pages_pdf, value=1)
    with col2:
        end_page = st.number_input("End Page", min_value=1, max_value=total_pages_pdf, value=total_pages_pdf)
        
    enable_zipper = st.checkbox(
        "Enable Multi-line Text Zipping", 
        value=False, 
        help="Enable this only if your PDF contains long text descriptions that wrap onto multiple lines within a single cell."
    )
        
    if st.button("Extract and Process Data"):
        if start_page > end_page:
            st.error("Start Page cannot be greater than End Page.")
        else:
            total_to_process = end_page - start_page + 1
            
            progress_text = st.empty()
            progress_bar = st.progress(0.0)
            
            baseline_seconds_per_page = 0.12 
            initial_eta = total_to_process * baseline_seconds_per_page
            
            progress_text.markdown(
                f"⏳ **Initializing PDFPlumber engine...** | "
                f"Estimated remaining time: **{initial_eta:.1f}s**"
            )
            
            all_extracted_dfs = []
            start_overall_time = time.time()
            last_ui_update_time = 0
            
            try:
                with pdfplumber.open(temp_pdf_path) as pdf:
                    for i, page_num in enumerate(range(start_page, end_page + 1)):
                        page = pdf.pages[page_num - 1]
                        
                        elapsed = time.time() - start_overall_time
                        pages_processed = i
                        
                        if pages_processed > 0:
                            avg_time_per_page = elapsed / pages_processed
                        else:
                            avg_time_per_page = baseline_seconds_per_page
                            
                        remaining_pages = total_to_process - pages_processed
                        current_eta = remaining_pages * avg_time_per_page
                        
                        current_time = time.time()
                        if current_time - last_ui_update_time > 0.4 or i == total_to_process - 1:
                            progress_text.markdown(
                                f"⏳ **Extracting page {page_num} of {end_page}** | "
                                f"Elapsed: **{elapsed:.1f}s** | "
                                f"Estimated remaining: **{current_eta:.1f}s**"
                            )
                            progress_bar.progress(min((i + 1) / total_to_process, 1.0))
                            last_ui_update_time = current_time
                        
                        tables = page.extract_tables(table_settings={
                            "vertical_strategy": "lines",
                            "horizontal_strategy": "lines",
                            "snap_tolerance": 3,
                            "join_tolerance": 3,
                        })
                        
                        for table in tables:
                            if not table:
                                continue
                            df = pd.DataFrame(table)
                            df = df.replace(to_replace=[r'\r', r'\n'], value=' ', regex=True)
                            
                            df = align_table_header(df)
                            
                            if not df.empty and len(df) > 1:
                                all_extracted_dfs.append(df)
                                
                if not all_extracted_dfs:
                    progress_text.empty()
                    progress_bar.empty()
                    st.error("No tables found with structural grid lines in the selected page range.")
                else:
                    progress_text.markdown(
                        f"✅ **Extraction complete!** | Total elapsed time: **{time.time() - start_overall_time:.1f}s**"
                    )
                    
                    # Grouping & Routing Phase
                    grouped_tables = {}
                    active_schemas = {}
                    
                    for df in all_extracted_dfs:
                        col_count = len(df.columns)
                        first_row = df.iloc[0]
                        
                        if is_likely_header(first_row):
                            schema_key = tuple(normalize_header_row(first_row))
                            active_schemas[col_count] = schema_key
                        else:
                            schema_key = active_schemas.get(col_count)
                            if not schema_key:
                                schema_key = tuple(f"col_{i}" for i in range(col_count))
                                active_schemas[col_count] = schema_key
                                
                        if schema_key not in grouped_tables:
                            grouped_tables[schema_key] = []
                        grouped_tables[schema_key].append(df)
                    
                    output_buffer = io.BytesIO()
                    
                    with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
                        for idx, (schema_key, df_list) in enumerate(grouped_tables.items(), start=1):
                            master_df = df_list[0]
                            
                            if len(df_list) > 1:
                                for next_df in df_list[1:]:
                                    if are_headers_matching(master_df.iloc[0], next_df.iloc[0]):
                                        next_df = next_df.iloc[1:].reset_index(drop=True)
                                    master_df = pd.concat([master_df, next_df], ignore_index=True)
                                    
                            if master_df.empty:
                                continue
                            
                            desc_idx, anchor_idx = detect_column_indices(master_df, master_df.iloc[0])
                            
                            master_df = clean_ghost_rows_safe(master_df, desc_col_idx=desc_idx, anchor_col_idx=anchor_idx)
                            
                            if enable_zipper:
                                master_df = zip_wrapped_text_optimized(master_df, desc_col_idx=desc_idx, anchor_col_idx=anchor_idx)
                            
                            master_df = remove_duplicate_headers(master_df, master_df.iloc[0])
                            
                            # Clean raw strings and restore native date formats [1]
                            master_df = clean_raw_strings_and_type_dates(master_df)
                            
                            sheet_name = f"Layout_Group_{idx}"
                            master_df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
                    
                    excel_data = output_buffer.getvalue()
                    
                    st.balloons()
                    st.success("Conversion complete!")
                    
                    st.download_button(
                        label="📥 Download Excel Spreadsheet",
                        data=excel_data,
                        file_name="Converted_Tables.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            
            except Exception as e:
                st.error(f"Processing failed: {e}")
                
            finally:
                if os.path.exists(temp_pdf_path):
                    os.unlink(temp_pdf_path)

# ==========================================
# 5. SYSTEM REQUIREMENTS (FOOTER)
# ==========================================
st.markdown("---")
st.caption("⚙️ Pure Python extraction engine. No Ghostscript required.")
