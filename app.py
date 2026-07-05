"""
Solar Tender Intelligence — AI-powered Solar RFP and tender document analyzer.
Powered by BAESS.APP
"""

import io
from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

from analysis import is_date_urgent, run_all_analyses
from config import COLORS, NVIDIA_DEEPSEEK_API_KEY, NVIDIA_MINIMAX_API_KEY
from export import generate_excel_checklist, generate_pdf_report
from pdf_extraction import detect_pdf_type, extract_document, is_solar_related
from rag import RAGPipeline
from styles import inject_custom_css

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Solar Tender Intelligence — BAESS.APP",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(inject_custom_css(), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def reset_session_state():
    """Clear all document and analysis state."""
    keys_to_clear = [
        "pdf_bytes", "doc_name", "routing_mode", "page_count", "avg_chars",
        "document_text", "rag", "chunk_count", "analysis", "routing_banner",
        "checklist_state", "eligibility_state", "processed", "processing_error",
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state["checklist_state"] = {}
    st.session_state["eligibility_state"] = {}


def init_session_state():
    defaults = {
        "pdf_bytes": None,
        "doc_name": "",
        "routing_mode": "text",
        "page_count": 0,
        "avg_chars": 0.0,
        "document_text": "",
        "rag": None,
        "chunk_count": 0,
        "analysis": None,
        "routing_banner": "",
        "checklist_state": {},
        "eligibility_state": {},
        "processed": False,
        "processing_error": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def process_document(pdf_bytes: bytes, doc_name: str):
    """Full pipeline: detect → extract → embed → analyze."""
    st.session_state["processing_error"] = None

    # Validate API keys based on routing (we detect first)
    try:
        routing_mode, page_count, avg_chars = detect_pdf_type(pdf_bytes)
    except Exception as e:
        st.session_state["processing_error"] = f"Failed to read PDF: {e}"
        return

    st.session_state["routing_mode"] = routing_mode
    st.session_state["page_count"] = page_count
    st.session_state["avg_chars"] = avg_chars
    st.session_state["doc_name"] = doc_name
    st.session_state["pdf_bytes"] = pdf_bytes

    if routing_mode == "scanned":
        if not NVIDIA_MINIMAX_API_KEY:
            st.session_state["processing_error"] = (
                "Scanned PDF detected but NVIDIA_MINIMAX_API_KEY is missing. "
                "Add it to your .env file or Streamlit Secrets."
            )
            return
        st.session_state["routing_banner"] = (
            "Scanned PDF detected — using MiniMax-M3 (NVIDIA NIM) for extraction"
        )
    else:
        if not NVIDIA_DEEPSEEK_API_KEY:
            st.session_state["processing_error"] = (
                "Text PDF detected but NVIDIA_DEEPSEEK_API_KEY is missing. "
                "Add it to your .env file or Streamlit Secrets."
            )
            return
        st.session_state["routing_banner"] = (
            "Text PDF detected — using DeepSeek-V4-Flash (NVIDIA NIM) for analysis"
        )

    progress = st.progress(0, text="Starting document processing…")

    try:
        # Extraction
        progress.progress(10, text="Extracting text from PDF…")
        document_text = extract_document(pdf_bytes, routing_mode)

        st.session_state["document_text"] = document_text
        progress.progress(35, text="Text extraction complete. Building search index…")

        if not is_solar_related(document_text):
            st.session_state["processing_error"] = (
                "Warning: This document may not contain solar tender content. "
                "Analysis will proceed but results may be limited."
            )

        # RAG
        rag = RAGPipeline()
        chunk_count = rag.build(document_text)
        st.session_state["rag"] = rag
        st.session_state["chunk_count"] = chunk_count
        progress.progress(55, text=f"Indexed {chunk_count} document chunks. Running AI analysis…")

        if chunk_count == 0:
            st.session_state["processing_error"] = "No extractable text found in this document."
            progress.empty()
            return

        # Analysis — all tabs at once
        section_labels = {
            "summary": "Document Summary",
            "technical": "Technical Requirements",
            "procedure": "Tender Procedure",
            "checklist": "Submission Checklist",
            "dates": "Key Dates",
            "commercial": "Commercial & Risk",
            "eligibility": "Eligibility & Deviations",
        }

        def analysis_progress(current, total, section_key):
            pct = 55 + int((current / total) * 40)
            progress.progress(
                pct,
                text=f"Analyzing: {section_labels.get(section_key, section_key)} ({current + 1}/{total})…",
            )

        analysis = run_all_analyses(rag, routing_mode, progress_callback=analysis_progress)
        st.session_state["analysis"] = analysis
        st.session_state["processed"] = True
        st.session_state["checklist_state"] = {}
        st.session_state["eligibility_state"] = {}

        progress.progress(100, text="Analysis complete!")
        progress.empty()

    except Exception as e:
        err_msg = str(e)
        if "504" in err_msg or "gateway" in err_msg.lower() or "timeout" in err_msg.lower():
            st.session_state["processing_error"] = (
                "The AI API timed out (504). This can happen on large tenders. "
                "The app now retries automatically — please click **Re-analyze** to try again. "
                f"Details: {err_msg}"
            )
        else:
            st.session_state["processing_error"] = f"Processing failed: {err_msg}"
        progress.empty()


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------

def render_header():
    st.markdown(
        """
        <div class="main-header">
            <h1>⚡ Solar Tender Intelligence</h1>
            <p>AI-powered Solar RFP &amp; Tender Document Analyzer — Powered by BAESS.APP</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_upload_prompt():
    st.markdown(
        """
        <div class="upload-prompt">
            <h2>📄 Upload a Solar Tender Document</h2>
            <p>Upload a PDF tender document (NIT, RFP, or EPC bid document) to begin AI-powered analysis.</p>
            <p>The system automatically detects scanned vs. text PDFs and routes to the optimal AI model.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_summary_tab(summary: dict):
    st.markdown('<div class="section-title">Executive Summary</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    if "raw_text" in summary:
        st.markdown(summary["raw_text"])
        st.markdown("</div>", unsafe_allow_html=True)
        return

    cols = st.columns(2)
    fields = [
        ("Project Title", summary.get("project_title")),
        ("Tendering Authority", summary.get("tendering_authority")),
        ("Project Capacity", summary.get("project_capacity")),
        ("Project Location", summary.get("project_location")),
        ("Technology Type", summary.get("technology_type")),
        ("Tender Type", summary.get("tender_type")),
        ("Estimated Project Value", summary.get("estimated_project_value")),
        ("Contract Duration", summary.get("contract_duration")),
    ]
    for i, (label, value) in enumerate(fields):
        with cols[i % 2]:
            st.metric(label, value or "Not specified")

    st.markdown("**Scope Summary**")
    st.write(summary.get("scope_summary", "Not specified in document."))
    st.markdown("</div>", unsafe_allow_html=True)


def render_technical_tab(content: str):
    st.markdown('<div class="section-title">Technical Requirements</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(content or "No technical requirements extracted.")
    st.markdown("</div>", unsafe_allow_html=True)


def render_procedure_tab(content: str):
    st.markdown('<div class="section-title">Tender Documentation Procedure</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(content or "No procedure information extracted.")
    st.markdown("</div>", unsafe_allow_html=True)


def render_checklist_tab(checklist: dict):
    st.markdown('<div class="section-title">Document Submission Checklist</div>', unsafe_allow_html=True)
    st.caption("Check off items as you prepare your bid submission.")

    if "raw_text" in checklist:
        st.markdown(checklist["raw_text"])
        return

    if not isinstance(checklist, dict):
        st.warning("Could not parse checklist data.")
        return

    for category, items in checklist.items():
        if category == "raw_text":
            continue
        st.markdown(f"**{category}**")
        if not isinstance(items, list):
            continue
        for idx, item in enumerate(items):
            key = f"chk_{category}_{idx}"
            if key not in st.session_state["checklist_state"]:
                st.session_state["checklist_state"][key] = False
            st.session_state["checklist_state"][key] = st.checkbox(
                str(item),
                value=st.session_state["checklist_state"][key],
                key=key,
            )

    total = len(st.session_state["checklist_state"])
    done = sum(1 for v in st.session_state["checklist_state"].values() if v)
    if total:
        st.progress(done / total, text=f"Progress: {done}/{total} documents prepared")


def render_dates_tab(dates_data: dict):
    st.markdown('<div class="section-title">Key Dates & Milestones</div>', unsafe_allow_html=True)

    dates_list = dates_data.get("dates", []) if isinstance(dates_data, dict) else []
    if not dates_list:
        st.info("No dates extracted from the document.")
        return

    cols = st.columns(3)
    for i, entry in enumerate(dates_list):
        if not isinstance(entry, dict):
            continue
        label = entry.get("label", "Date")
        date_val = entry.get("date", "Not specified")
        time_val = entry.get("time", "")
        venue = entry.get("venue", "")

        urgent = is_date_urgent(str(date_val))
        with cols[i % 3]:
            display = str(date_val)
            if time_val and str(time_val).lower() not in ("not specified", "", "none"):
                display += f" {time_val}"
            st.metric(
                label=f"{label}{' ⚠️ URGENT' if urgent else ''}",
                value=display,
            )
            if venue and str(venue).lower() not in ("not specified", "", "none"):
                st.caption(f"📍 {venue}")


def render_commercial_tab(commercial_data: dict):
    st.markdown('<div class="section-title">Commercial Summary</div>', unsafe_allow_html=True)

    if "raw_text" in commercial_data:
        st.markdown(commercial_data["raw_text"])
        return

    comm = commercial_data.get("commercial", {}) if isinstance(commercial_data, dict) else {}
    if comm:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        fields = list(comm.items())
        mid = (len(fields) + 1) // 2
        with c1:
            for key, val in fields[:mid]:
                st.markdown(f"**{key.replace('_', ' ').title()}:** {val or 'Not specified'}")
        with c2:
            for key, val in fields[mid:]:
                st.markdown(f"**{key.replace('_', ' ').title()}:** {val or 'Not specified'}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Risk Flags</div>', unsafe_allow_html=True)
    risks = commercial_data.get("risks", []) if isinstance(commercial_data, dict) else []
    if not risks:
        st.info("No specific risks identified.")
    for risk in risks:
        if not isinstance(risk, dict):
            continue
        level = str(risk.get("level", "minor")).lower()
        title = risk.get("title", "Risk")
        desc = risk.get("description", "")
        msg = f"**{title}** — {desc}"
        if level == "critical":
            st.error(msg)
        elif level == "moderate":
            st.warning(msg)
        else:
            st.info(msg)

    st.markdown('<div class="section-title">Bid / No-Bid Recommendation</div>', unsafe_allow_html=True)
    scoring = commercial_data.get("bid_scoring", {}) if isinstance(commercial_data, dict) else {}
    if scoring:
        params = scoring.get("parameters", [])
        if params:
            df = pd.DataFrame([
                {
                    "Parameter": p.get("name", ""),
                    "Score": p.get("score", ""),
                    "Max": p.get("max", 5),
                    "Notes": p.get("notes", ""),
                }
                for p in params if isinstance(p, dict)
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)

        total = scoring.get("total_score", "N/A")
        max_score = scoring.get("max_score", "N/A")
        rec = scoring.get("recommendation", "N/A")
        color = str(scoring.get("recommendation_color", "amber")).lower()
        css_class = {"green": "recommendation-green", "amber": "recommendation-amber", "red": "recommendation-red"}.get(
            color, "recommendation-amber"
        )
        st.markdown(
            f'<div class="{css_class}">Recommendation: {rec} | Score: {total}/{max_score}</div>',
            unsafe_allow_html=True,
        )
        st.write(scoring.get("rationale", ""))


def render_eligibility_tab(elig_data: dict):
    st.markdown('<div class="section-title">Eligibility Criteria</div>', unsafe_allow_html=True)

    if "raw_text" in elig_data:
        st.markdown(elig_data["raw_text"])
        return

    criteria = elig_data.get("eligibility", []) if isinstance(elig_data, dict) else []
    pass_count = 0
    fail_count = 0
    unknown_count = 0

    for idx, crit in enumerate(criteria):
        if not isinstance(crit, dict):
            continue
        key = f"elig_{idx}"
        st.markdown(f"**{crit.get('criterion', 'Criterion')}**")
        st.caption(f"Required: {crit.get('requirement', 'Not specified')} | Category: {crit.get('category', 'general')}")

        meets = st.radio(
            "Status",
            ["Meets", "Does Not Meet", "Uncertain"],
            key=f"{key}_radio",
            horizontal=True,
        )
        st.session_state["eligibility_state"][key] = meets
        if meets == "Meets":
            pass_count += 1
        elif meets == "Does Not Meet":
            fail_count += 1
        else:
            unknown_count += 1
        st.divider()

    total = len(criteria)
    if total:
        st.markdown('<div class="section-title">Eligibility Summary</div>', unsafe_allow_html=True)
        ec1, ec2, ec3 = st.columns(3)
        ec1.metric("Meets Criteria", pass_count)
        ec2.metric("Does Not Meet", fail_count)
        ec3.metric("Uncertain", unknown_count)

        if fail_count > 0:
            st.error(f"Eligibility FAIL — {fail_count} criterion/criteria not met.")
        elif unknown_count > 0:
            st.warning(f"Eligibility UNCERTAIN — {unknown_count} criterion/criteria need verification.")
        else:
            st.success(f"Eligibility PASS — All {pass_count} criteria appear to be met.")

    st.markdown('<div class="section-title">Technical Deviations from Standards</div>', unsafe_allow_html=True)
    devs = elig_data.get("deviations", []) if isinstance(elig_data, dict) else []
    if not devs:
        st.info("No material deviations from IS/IEC standards identified.")
    else:
        df = pd.DataFrame([
            {
                "Tender Specification": d.get("tender_spec", ""),
                "Applicable Standard": d.get("applicable_standard", ""),
                "Nature of Deviation": d.get("deviation_nature", ""),
            }
            for d in devs if isinstance(d, dict)
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)


def render_export_tab(analysis: dict, doc_name: str):
    st.markdown('<div class="section-title">Export Analysis</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Generate Full Report (PDF)**")
        st.caption("Compiles all analysis sections into a formatted PDF with BAESS.APP branding.")
        if st.button("Generate Full Report", type="primary", key="gen_pdf"):
            with st.spinner("Generating PDF report…"):
                try:
                    pdf_bytes = generate_pdf_report(doc_name, analysis)
                    st.session_state["pdf_report_bytes"] = pdf_bytes
                    st.success("PDF report generated!")
                except Exception as e:
                    st.error(f"PDF generation failed: {e}")

        if "pdf_report_bytes" in st.session_state:
            st.download_button(
                label="⬇️ Download PDF Report",
                data=st.session_state["pdf_report_bytes"],
                file_name=f"solar_tender_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
            )

    with col2:
        st.markdown("**Export Checklist to Excel**")
        st.caption("Structured workbook with one sheet per checklist category.")
        checklist = analysis.get("checklist", {})
        if st.button("Export Checklist to Excel", type="primary", key="gen_xlsx"):
            with st.spinner("Generating Excel workbook…"):
                try:
                    xlsx_bytes = generate_excel_checklist(checklist)
                    st.session_state["xlsx_bytes"] = xlsx_bytes
                    st.success("Excel checklist generated!")
                except Exception as e:
                    st.error(f"Excel export failed: {e}")

        if "xlsx_bytes" in st.session_state:
            st.download_button(
                label="⬇️ Download Excel Checklist",
                data=st.session_state["xlsx_bytes"],
                file_name=f"solar_tender_checklist_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    init_session_state()
    render_header()

    # ---- Sidebar ----
    with st.sidebar:
        st.markdown("### 📁 Document Upload")
        uploaded = st.file_uploader(
            "Upload Tender PDF",
            type=["pdf"],
            help="Upload NIT, RFP, or EPC tender document in PDF format.",
        )

        if uploaded is not None:
            pdf_bytes = uploaded.read()
            if st.session_state.get("doc_name") != uploaded.name or not st.session_state.get("processed"):
                with st.spinner("Processing document…"):
                    process_document(pdf_bytes, uploaded.name)

        if st.session_state.get("routing_banner"):
            st.info(st.session_state["routing_banner"])

        if st.session_state.get("processing_error"):
            err = st.session_state["processing_error"]
            if err.startswith("Warning:"):
                st.warning(err)
            else:
                st.error(err)

        if st.session_state.get("processed"):
            st.markdown("---")
            st.markdown("### 📊 Document Metadata")
            st.markdown(f"**Name:** {st.session_state['doc_name']}")
            st.markdown(f"**Pages:** {st.session_state['page_count']}")
            mode_label = (
                "Scanned (MiniMax-M3)"
                if st.session_state["routing_mode"] == "scanned"
                else "Text (DeepSeek-V4-Flash)"
            )
            st.markdown(f"**Detected Type:** {mode_label}")
            st.markdown(f"**Avg Chars/Page:** {st.session_state['avg_chars']:.0f}")
            st.markdown(f"**RAG Chunks:** {st.session_state['chunk_count']}")
            st.markdown(f"**Extracted Text:** {len(st.session_state.get('document_text', '')):,} chars")

            if st.button("🔄 Re-analyze", use_container_width=True):
                if st.session_state.get("pdf_bytes"):
                    with st.spinner("Re-running analysis…"):
                        process_document(st.session_state["pdf_bytes"], st.session_state["doc_name"])
                    st.rerun()

            if st.button("🗑️ Clear & Upload New Document", use_container_width=True):
                reset_session_state()
                st.rerun()

        st.markdown("---")
        st.markdown(
            f'<p style="color:{COLORS["slate"]};font-size:0.75rem;">'
            "Solar Tender Intelligence v1.0<br>Powered by BAESS.APP</p>",
            unsafe_allow_html=True,
        )

    # ---- Main content ----
    if not st.session_state.get("processed") or not st.session_state.get("analysis"):
        render_upload_prompt()
        st.markdown("#### How it works")
        steps = st.columns(4)
        steps[0].markdown("**1. Upload**\n\nUpload your tender PDF")
        steps[1].markdown("**2. Detect**\n\nAuto-detect scanned vs text PDF")
        steps[2].markdown("**3. Analyze**\n\nRAG + AI extracts 8 analysis views")
        steps[3].markdown("**4. Export**\n\nDownload PDF report & Excel checklist")
        return

    analysis = st.session_state["analysis"]

    tabs = st.tabs([
        "📋 Summary",
        "⚙️ Technical",
        "📝 Procedure",
        "✅ Checklist",
        "📅 Dates",
        "💰 Commercial",
        "🔍 Eligibility",
        "📤 Export",
    ])

    with tabs[0]:
        render_summary_tab(analysis.get("summary", {}))
    with tabs[1]:
        render_technical_tab(analysis.get("technical", ""))
    with tabs[2]:
        render_procedure_tab(analysis.get("procedure", ""))
    with tabs[3]:
        render_checklist_tab(analysis.get("checklist", {}))
    with tabs[4]:
        render_dates_tab(analysis.get("dates", {}))
    with tabs[5]:
        render_commercial_tab(analysis.get("commercial", {}))
    with tabs[6]:
        render_eligibility_tab(analysis.get("eligibility", {}))
    with tabs[7]:
        render_export_tab(analysis, st.session_state["doc_name"])

    st.markdown(
        '<p class="baess-watermark">Solar Tender Intelligence — Powered by BAESS.APP</p>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
