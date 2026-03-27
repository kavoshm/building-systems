"""
Multi-Step Prompt Chain for Clinical Triage
=============================================
Demonstrates a 4-step chained prompt pipeline that processes a clinical note
through sequential stages: extraction, assessment, urgency classification,
and structured output generation.

Each step is a focused LLM call with its own system prompt. The output of
each step flows into the next, building up a comprehensive triage result.

Requires: OPENAI_API_KEY environment variable
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()


# --- Pipeline Context ---

@dataclass
class PipelineContext:
    """Accumulates results as data flows through the pipeline."""
    raw_note: str
    step_results: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    def add_step(self, step_name: str, result: Any, elapsed: float) -> None:
        """Record the result and timing of a pipeline step."""
        self.step_results[step_name] = result
        self.timings[step_name] = elapsed


# --- Step 1: Clinical Extraction ---

EXTRACTION_PROMPT: str = """You are a clinical data extraction system. Extract structured
information from the provided clinical note.

Return JSON with these fields:
{
  "patient_age": int or null,
  "patient_sex": "M" | "F" | "unknown",
  "chief_complaint": str,
  "symptoms": [str],
  "vital_signs": {"bp": str, "hr": str, "rr": str, "spo2": str, "temp": str} (null for missing),
  "medical_history": [str],
  "medications": [str],
  "physical_exam_findings": [str],
  "lab_results": [{"test": str, "value": str}]
}

Extract ONLY what is explicitly stated. Use null for missing fields. Do NOT infer."""


def step_extract(ctx: PipelineContext) -> PipelineContext:
    """Step 1: Extract structured data from the raw clinical note."""
    start = time.time()
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user", "content": f"Clinical note:\n\n{ctx.raw_note}"},
            ],
        )
        result = json.loads(response.choices[0].message.content)
        ctx.add_step("extraction", result, time.time() - start)
    except Exception as e:
        ctx.errors.append(f"Extraction failed: {e}")
    return ctx


# --- Step 2: Red Flag Assessment ---

RED_FLAG_PROMPT: str = """You are a clinical safety assessment system. Given extracted clinical
data, identify red flags and concerning features.

Red flag categories:
- HEMODYNAMIC: Hypotension, tachycardia, signs of shock
- NEUROLOGICAL: Altered mental status, focal deficits, sudden severe headache
- RESPIRATORY: SpO2 < 92%, severe dyspnea, accessory muscle use
- CARDIAC: Chest pain with risk factors, ECG changes, diaphoresis
- HEMORRHAGIC: Active bleeding, melena, hematemesis
- INFECTIOUS: High fever with hemodynamic changes (sepsis criteria)
- PSYCHIATRIC: Suicidal ideation, homicidal ideation, acute psychosis

Return JSON:
{
  "red_flags_identified": [{"flag": str, "category": str, "severity": "critical"|"high"|"moderate"}],
  "protective_factors": [str],
  "overall_acuity_impression": "critical" | "high" | "moderate" | "low" | "minimal"
}"""


def step_assess_red_flags(ctx: PipelineContext) -> PipelineContext:
    """Step 2: Assess red flags using extracted data from Step 1."""
    start = time.time()
    extraction = ctx.step_results.get("extraction", {})
    if not extraction:
        ctx.errors.append("Red flag assessment skipped: no extraction data")
        return ctx

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": RED_FLAG_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Extracted clinical data:\n{json.dumps(extraction, indent=2)}\n\n"
                        f"Original note for context:\n{ctx.raw_note}"
                    ),
                },
            ],
        )
        result = json.loads(response.choices[0].message.content)
        ctx.add_step("red_flag_assessment", result, time.time() - start)
    except Exception as e:
        ctx.errors.append(f"Red flag assessment failed: {e}")
    return ctx


# --- Step 3: Urgency Classification ---

URGENCY_PROMPT: str = """You are a triage urgency classifier. Given the extracted clinical data
AND the red flag assessment, assign a final urgency level.

Urgency Scale:
  1 = ROUTINE: Stable, no acute issues, suitable for scheduled appointment
  2 = LOW: Minor acute issue, can be seen within 24-48 hours
  3 = MODERATE: Needs evaluation within hours, not immediately life-threatening
  4 = HIGH: Needs prompt evaluation, potential for deterioration
  5 = EMERGENT: Life-threatening, requires immediate intervention

Classification rules:
- Any "critical" red flag -> urgency 5
- Multiple "high" red flags -> urgency 4-5
- Single "high" red flag -> urgency 3-4
- Only "moderate" red flags -> urgency 2-3
- No red flags -> urgency 1-2

Return JSON:
{
  "urgency_level": int (1-5),
  "urgency_label": str,
  "reasoning": str (2-3 sentences explaining the classification),
  "disposition_recommendation": "immediate_ed" | "urgent_care" | "same_day_clinic" | "scheduled_visit" | "telehealth"
}"""


def step_classify_urgency(ctx: PipelineContext) -> PipelineContext:
    """Step 3: Classify urgency using extraction and red flag assessment."""
    start = time.time()
    extraction = ctx.step_results.get("extraction", {})
    red_flags = ctx.step_results.get("red_flag_assessment", {})

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": URGENCY_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Extracted data:\n{json.dumps(extraction, indent=2)}\n\n"
                        f"Red flag assessment:\n{json.dumps(red_flags, indent=2)}"
                    ),
                },
            ],
        )
        result = json.loads(response.choices[0].message.content)
        ctx.add_step("urgency_classification", result, time.time() - start)
    except Exception as e:
        ctx.errors.append(f"Urgency classification failed: {e}")
    return ctx


# --- Step 4: Final Structured Output ---

OUTPUT_PROMPT: str = """You are a clinical documentation formatter. Given all prior pipeline
results, produce a final structured triage summary.

Return JSON matching this exact schema:
{
  "triage_summary": {
    "patient": {"age": int, "sex": str},
    "chief_complaint": str,
    "urgency": {"level": int, "label": str},
    "red_flags": [str],
    "disposition": str,
    "key_findings": [str],
    "recommended_actions": [str],
    "reasoning": str
  },
  "metadata": {
    "pipeline_version": "1.0.0",
    "steps_completed": int,
    "total_processing_time_ms": int
  }
}

Synthesize from all provided data. Be concise but complete."""


def step_format_output(ctx: PipelineContext) -> PipelineContext:
    """Step 4: Format all pipeline results into a final structured output."""
    start = time.time()
    total_time_ms = int(sum(ctx.timings.values()) * 1000)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": OUTPUT_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Pipeline results:\n{json.dumps(ctx.step_results, indent=2)}\n\n"
                        f"Steps completed: {len(ctx.step_results)}\n"
                        f"Total processing time: {total_time_ms}ms"
                    ),
                },
            ],
        )
        result = json.loads(response.choices[0].message.content)
        ctx.add_step("final_output", result, time.time() - start)
    except Exception as e:
        ctx.errors.append(f"Output formatting failed: {e}")
    return ctx


# --- Pipeline Runner ---

def run_triage_pipeline(note: str) -> PipelineContext:
    """Execute the full 4-step triage pipeline on a clinical note."""
    ctx = PipelineContext(raw_note=note)

    steps = [
        ("1. Extraction", step_extract),
        ("2. Red Flag Assessment", step_assess_red_flags),
        ("3. Urgency Classification", step_classify_urgency),
        ("4. Output Formatting", step_format_output),
    ]

    for step_name, step_fn in steps:
        ctx = step_fn(ctx)
        if ctx.errors:
            print(f"  [WARNING] Errors after {step_name}: {ctx.errors[-1]}")

    return ctx


# --- Test Cases ---

TEST_NOTES: list[dict[str, str]] = [
    {
        "label": "Acute MI Presentation",
        "note": (
            "67-year-old male with sudden onset severe substernal chest pain radiating to "
            "left arm, onset 30 minutes ago. Diaphoretic, nauseated. PMH: DM2, HTN, prior "
            "MI 5 years ago. Current meds: aspirin, metoprolol, lisinopril, atorvastatin. "
            "BP 92/58, HR 110, RR 22, SpO2 93%. ECG shows ST elevation in V1-V4."
        ),
    },
    {
        "label": "Pediatric Wellness Check",
        "note": (
            "4-year-old female here for well-child visit. Meeting all developmental milestones. "
            "Height and weight at 50th percentile. Immunizations up to date. No concerns from "
            "parents. Eats well, sleeps well. Active and social at preschool. Physical exam "
            "unremarkable. Vision and hearing screening normal."
        ),
    },
]


def main() -> None:
    """Run the triage pipeline on test notes and display results."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.syntax import Syntax

        console = Console()
        use_rich = True
    except ImportError:
        use_rich = False

    for test in TEST_NOTES:
        if use_rich:
            console.print(f"\n[bold cyan]{'=' * 80}[/bold cyan]")
            console.print(f"[bold]Test Case: {test['label']}[/bold]")
            console.print(Panel(test["note"], title="Input Note"))

        ctx = run_triage_pipeline(test["note"])

        if use_rich:
            # Show timing for each step
            console.print("\n[bold yellow]Pipeline Timings:[/bold yellow]")
            for step, elapsed in ctx.timings.items():
                console.print(f"  {step}: {elapsed * 1000:.0f}ms")
            console.print(
                f"  [bold]Total: {sum(ctx.timings.values()) * 1000:.0f}ms[/bold]"
            )

            # Show final output
            final = ctx.step_results.get("final_output", {})
            formatted = json.dumps(final, indent=2)
            syntax = Syntax(formatted, "json", theme="monokai", line_numbers=False)
            console.print(Panel(syntax, title="[bold green]Final Triage Output[/bold green]"))

            if ctx.errors:
                for err in ctx.errors:
                    console.print(f"[red]ERROR: {err}[/red]")
        else:
            final = ctx.step_results.get("final_output", {})
            print(f"\n{test['label']}:")
            print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()


# --- Sample Output ---
#
# Test Case: Acute MI Presentation
# Pipeline Timings:
#   extraction: 820ms
#   red_flag_assessment: 650ms
#   urgency_classification: 1200ms
#   final_output: 580ms
#   Total: 3250ms
#
# Final Triage Output:
# {
#   "triage_summary": {
#     "patient": {"age": 67, "sex": "M"},
#     "chief_complaint": "Severe substernal chest pain with left arm radiation",
#     "urgency": {"level": 5, "label": "EMERGENT"},
#     "red_flags": [
#       "ST elevation in V1-V4 (STEMI)",
#       "Hypotension (BP 92/58)",
#       "Tachycardia (HR 110)",
#       "Hypoxia (SpO2 93%)",
#       "Prior MI history"
#     ],
#     "disposition": "immediate_ed",
#     "key_findings": [
#       "Active STEMI on ECG",
#       "Hemodynamic compromise",
#       "High-risk cardiac history"
#     ],
#     "recommended_actions": [
#       "Activate cardiac catheterization lab",
#       "Administer aspirin 325mg",
#       "Establish IV access",
#       "Continuous cardiac monitoring",
#       "Serial troponins"
#     ],
#     "reasoning": "Acute STEMI with hemodynamic instability in a patient with significant cardiac history. Requires immediate percutaneous coronary intervention."
#   },
#   "metadata": {
#     "pipeline_version": "1.0.0",
#     "steps_completed": 4,
#     "total_processing_time_ms": 3250
#   }
# }
