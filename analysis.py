"""Analysis prompts and orchestration for all tabs."""

from datetime import datetime
from typing import Any

from config import RETRIEVE_K
from llm import call_llm, parse_json_response
from rag import RAGPipeline

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SUMMARY_PROMPT = """Analyze the tender and return a JSON object with these exact keys:
{
  "project_title": "string",
  "tendering_authority": "string",
  "project_capacity": "string (MWp/MWh)",
  "project_location": "string",
  "technology_type": "ground-mount/rooftop/floating/hybrid/other",
  "tender_type": "open/limited/EPC/supply/other",
  "estimated_project_value": "string",
  "contract_duration": "string",
  "scope_summary": "one paragraph summary of project scope"
}
Return ONLY valid JSON."""

TECHNICAL_PROMPT = """Extract technical requirements from this solar tender for the specified sections only.
Use markdown bullet points with spec values where stated. Max 15 bullets per subsection.
State N/A if a subsection has no requirements in the excerpts."""

# Smaller batches — one API call each when running the Technical step
TECHNICAL_BATCHES = [
    {
        "title": "PV Modules & Inverters",
        "query": "solar PV modules efficiency power tolerance IEC 61215 61730 BIS ALMM inverter grid LVRT HVRT",
        "sections": "## Solar PV Modules\n## Inverters",
    },
    {
        "title": "Mounting, Cabling & SCADA",
        "query": "mounting structure wind load cabling DC AC SCADA monitoring data logger transformer",
        "sections": "## Mounting Structure\n## DC and AC Cabling\n## Monitoring & SCADA\n## Transformers and HV Equipment",
    },
    {
        "title": "Civil, Earthing & Safety",
        "query": "civil works fencing road earthing lightning protection safety access lighting signage",
        "sections": "## Civil Works\n## Earthing and Lightning Protection\n## Safety and Access",
    },
    {
        "title": "Grid & Project-Specific",
        "query": "grid interconnection metering protection relay export limitation project specific technical requirements",
        "sections": "## Grid Interconnection\n## Project-Specific Requirements",
    },
]

PROCEDURE_PROMPT = """Produce a numbered step-by-step guide for preparing and submitting the bid for this specific solar EPC tender in India.
Cover all of these steps (tailor to document specifics):
1. Bid document procurement and portal registration
2. Studying NIT, scope, BOQ, SCC/GCC
3. Site visit and reconnaissance
4. Pre-bid meeting and query submission
5. Corrigendum tracking
6. Technical bid preparation (drawings, datasheets, OEM auth, deviations)
7. Commercial bid preparation (price schedule, taxes, price variation)
8. Financial and legal documentation
9. EMD arrangement
10. Portal upload sequencing
11. Physical submission if required
12. Bid opening attendance
13. Post-bid clarification process

Use markdown numbered list format."""

CHECKLIST_PROMPT = """Return a JSON object with checklist categories and items:
{
  "Technical Documents": ["item1", "item2", ...],
  "Financial Documents": ["item1", ...],
  "Legal and Statutory Documents": ["item1", ...],
  "Experience and Credential Documents": ["item1", ...],
  "Tender-Specific Forms": ["item1", ...],
  "Bid Security": ["item1", ...]
}
List every document required for bid submission found in the tender. Return ONLY valid JSON."""

DATES_PROMPT = """Extract all critical dates from the tender. Return JSON:
{
  "dates": [
    {"label": "Document Download Start", "date": "YYYY-MM-DD or text", "time": "optional", "venue": "optional"},
    {"label": "Document Download End", "date": "...", "time": "...", "venue": "..."},
    {"label": "Pre-bid Meeting", "date": "...", "time": "...", "venue": "..."},
    {"label": "Last Date for Written Queries", "date": "...", "time": "...", "venue": "..."},
    {"label": "Corrigendum Expected", "date": "...", "time": "...", "venue": "..."},
    {"label": "Bid Submission Deadline", "date": "...", "time": "...", "venue": "..."},
    {"label": "Bid Opening Date", "date": "...", "time": "...", "venue": "..."},
    {"label": "Technical Evaluation Period", "date": "...", "time": "...", "venue": "..."},
    {"label": "Financial Bid Opening", "date": "...", "time": "...", "venue": "..."},
    {"label": "LOA Issuance Estimate", "date": "...", "time": "...", "venue": "..."},
    {"label": "Work Order Date", "date": "...", "time": "...", "venue": "..."},
    {"label": "Project Completion Date", "date": "...", "time": "...", "venue": "..."},
    {"label": "O&M Period End", "date": "...", "time": "...", "venue": "..."}
  ]
}
Include only dates found or reasonably inferable. Use "Not specified" for missing dates. Return ONLY valid JSON."""

COMMERCIAL_PROMPT = """Analyze commercial and risk aspects. Return JSON:
{
  "commercial": {
    "estimated_project_value": "string",
    "emd_amount_and_form": "string",
    "performance_bank_guarantee": "string",
    "advance_payment_terms": "string",
    "payment_milestones": "string",
    "retention_money": "string",
    "price_variation_clause": "yes/no with formula if present",
    "arbitration_dispute": "string"
  },
  "risks": [
    {"level": "critical|moderate|minor", "title": "short title", "description": "detail"}
  ],
  "bid_scoring": {
    "parameters": [
      {"name": "parameter name", "score": 1-5, "max": 5, "notes": "brief"}
    ],
    "total_score": number,
    "max_score": number,
    "recommendation": "BID|CONDITIONAL BID|NO-BID",
    "recommendation_color": "green|amber|red",
    "rationale": "brief explanation"
  }
}
Include 5-7 scoring parameters. Return ONLY valid JSON."""

ELIGIBILITY_PROMPT = """Extract eligibility and deviation information. Return JSON:
{
  "eligibility": [
    {"criterion": "description", "requirement": "what is required", "category": "financial|technical|statutory"}
  ],
  "deviations": [
    {"tender_spec": "spec from tender", "applicable_standard": "IS/IEC standard", "deviation_nature": "description"}
  ]
}
Return ONLY valid JSON."""


# ---------------------------------------------------------------------------
# RAG query strings for retrieval
# ---------------------------------------------------------------------------

RETRIEVAL_QUERIES = {
    "summary": "project title tendering authority capacity location technology tender type value duration scope",
    "technical": "technical specifications modules inverters mounting cabling SCADA transformers civil earthing grid interconnection",
    "procedure": "bid submission process portal registration pre-bid meeting EMD technical commercial envelope",
    "checklist": "documents required submission checklist drawings certificates financial legal EMD",
    "dates": "important dates deadline pre-bid meeting bid opening completion milestones",
    "commercial": "EMD performance guarantee payment retention liquidated damages price variation arbitration commercial terms",
    "eligibility": "eligibility criteria turnover net worth experience deviation specifications standards",
}

PROMPTS = {
    "summary": SUMMARY_PROMPT,
    "technical": TECHNICAL_PROMPT,
    "procedure": PROCEDURE_PROMPT,
    "checklist": CHECKLIST_PROMPT,
    "dates": DATES_PROMPT,
    "commercial": COMMERCIAL_PROMPT,
    "eligibility": ELIGIBILITY_PROMPT,
}

# Ordered analysis steps — one API call each
ANALYSIS_SECTIONS = [
    {"key": "summary", "label": "Document Summary", "is_json": True},
    {"key": "technical", "label": "Technical Requirements (4 parts)", "is_json": False},
    {"key": "procedure", "label": "Tender Procedure", "is_json": False},
    {"key": "checklist", "label": "Submission Checklist", "is_json": True},
    {"key": "dates", "label": "Key Dates", "is_json": True},
    {"key": "commercial", "label": "Commercial & Risk", "is_json": True},
    {"key": "eligibility", "label": "Eligibility & Deviations", "is_json": True},
]


def run_technical_analysis(
    rag: RAGPipeline,
    routing_mode: str,
    progress_callback=None,
) -> str:
    """Run technical analysis as 4 smaller API calls to avoid timeouts."""
    import time

    from config import (
        API_CALL_DELAY_SECONDS,
        TECHNICAL_CONTEXT_CHARS,
        TECHNICAL_MAX_TOKENS,
        TECHNICAL_RETRIEVE_K,
    )

    parts: list[str] = []
    total = len(TECHNICAL_BATCHES)

    # Extra pause after summary step before heavy technical work
    time.sleep(API_CALL_DELAY_SECONDS)

    for i, batch in enumerate(TECHNICAL_BATCHES):
        if progress_callback:
            progress_callback(i, total, batch["title"])

        prompt = (
            f"{TECHNICAL_PROMPT}\n\n"
            f"Include ONLY these markdown subsections:\n{batch['sections']}"
        )
        chunks = rag.retrieve(batch["query"], k=TECHNICAL_RETRIEVE_K)
        raw = call_llm(
            prompt,
            chunks,
            routing_mode,
            max_tokens=TECHNICAL_MAX_TOKENS,
            max_context_chars=TECHNICAL_CONTEXT_CHARS,
            temperature=0.4,
        )
        parts.append(raw.strip())
        time.sleep(API_CALL_DELAY_SECONDS)

    return "\n\n".join(parts)


def get_combined_technical(analysis: dict) -> str:
    """Return technical content from analysis dict."""
    if not analysis:
        return ""
    tech = analysis.get("technical")
    if isinstance(tech, str) and tech.strip():
        return tech
    return ""


def run_section_analysis(
    rag: RAGPipeline,
    routing_mode: str,
    section_key: str,
    progress_callback=None,
) -> Any:
    """Run a single analysis section."""
    import time

    from config import API_CALL_DELAY_SECONDS

    if section_key == "technical":
        return run_technical_analysis(rag, routing_mode, progress_callback=progress_callback)

    if section_key not in PROMPTS:
        raise ValueError(f"Unknown analysis section: {section_key}")

    prompt = PROMPTS[section_key]
    query = RETRIEVAL_QUERIES[section_key]
    is_json = next(s["is_json"] for s in ANALYSIS_SECTIONS if s["key"] == section_key)

    chunks = rag.retrieve(query, k=RETRIEVE_K)
    raw = call_llm(prompt, chunks, routing_mode)
    result = parse_json_response(raw) if is_json else raw

    time.sleep(API_CALL_DELAY_SECONDS)
    return result


def run_all_analyses(
    rag: RAGPipeline,
    routing_mode: str,
    progress_callback=None,
) -> dict[str, Any]:
    """Run all sections sequentially (legacy helper — prefer run_section_analysis)."""
    results: dict[str, Any] = {}
    for i, section in enumerate(ANALYSIS_SECTIONS):
        key = section["key"]
        if progress_callback:
            progress_callback(i, len(ANALYSIS_SECTIONS), key)
        results[key] = run_section_analysis(rag, routing_mode, key)
    return results


def is_date_urgent(date_str: str) -> bool:
    """Check urgency with flexible parsing."""
    if not date_str:
        return False
    s = date_str.strip()
    for fmt in ["%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y"]:
        try:
            # Try full string and truncated
            for candidate in [s, s[:10]]:
                try:
                    d = datetime.strptime(candidate, fmt)
                    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                    return 0 <= (d - today).days <= 15
                except ValueError:
                    continue
        except ValueError:
            continue
    return False
