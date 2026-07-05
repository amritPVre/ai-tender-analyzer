"""Analysis prompts and orchestration for all tabs."""

from datetime import datetime
from typing import Any

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

TECHNICAL_PROMPT = """Extract ALL technical requirements from this solar tender document.
Organize into markdown with these sub-sections (include all that apply, note N/A if not found):
## Solar PV Modules
## Inverters
## Mounting Structure
## DC and AC Cabling
## Monitoring & SCADA
## Transformers and HV Equipment
## Civil Works
## Earthing and Lightning Protection
## Safety and Access
## Grid Interconnection
## Project-Specific Requirements

Each bullet must be a specific, actionable requirement with values or references from the document.
Be exhaustive — do not summarize or omit requirements."""

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


def run_all_analyses(
    rag: RAGPipeline,
    routing_mode: str,
    progress_callback=None,
) -> dict[str, Any]:
    """Run all analysis sections and return structured results."""
    results: dict[str, Any] = {}
    sections = [
        ("summary", SUMMARY_PROMPT, RETRIEVAL_QUERIES["summary"], 15, True),
        ("technical", TECHNICAL_PROMPT, RETRIEVAL_QUERIES["technical"], 20, False),
        ("procedure", PROCEDURE_PROMPT, RETRIEVAL_QUERIES["procedure"], 15, False),
        ("checklist", CHECKLIST_PROMPT, RETRIEVAL_QUERIES["checklist"], 15, True),
        ("dates", DATES_PROMPT, RETRIEVAL_QUERIES["dates"], 15, True),
        ("commercial", COMMERCIAL_PROMPT, RETRIEVAL_QUERIES["commercial"], 15, True),
        ("eligibility", ELIGIBILITY_PROMPT, RETRIEVAL_QUERIES["eligibility"], 15, True),
    ]

    for i, (key, prompt, query, k, is_json) in enumerate(sections):
        if progress_callback:
            progress_callback(i, len(sections), key)
        chunks = rag.retrieve(query, k=k)
        raw = call_llm(prompt, chunks, routing_mode)
        results[key] = parse_json_response(raw) if is_json else raw

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
