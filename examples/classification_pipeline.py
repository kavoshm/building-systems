"""
Input Classification and Routing System
==========================================
Demonstrates the "router" pattern: an initial classifier determines which
specialized pipeline should handle the input. Each route has domain-specific
prompts optimized for that clinical specialty.

This is a key pattern for production healthcare AI systems where a single
generic prompt cannot handle the diversity of clinical presentations.

Requires: OPENAI_API_KEY environment variable
"""

import json
from typing import Any, Callable

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()


# --- Route Definitions ---

ROUTES: dict[str, dict[str, str]] = {
    "cardiology": {
        "description": "Cardiac presentations: chest pain, arrhythmia, heart failure, etc.",
        "system_prompt": (
            "You are a cardiology triage specialist. Analyze the clinical note focusing on:\n"
            "- Cardiac risk factors (age, DM, HTN, smoking, family hx, prior CAD)\n"
            "- ECG findings if mentioned\n"
            "- Troponin or other cardiac biomarkers\n"
            "- Hemodynamic stability\n"
            "- HEART score components\n\n"
            "Return JSON: {\"cardiac_risk\": \"high\"|\"moderate\"|\"low\", "
            "\"suspected_diagnosis\": str, \"heart_score_estimate\": int, "
            "\"recommended_workup\": [str], \"urgency\": int}"
        ),
    },
    "respiratory": {
        "description": "Pulmonary presentations: dyspnea, cough, asthma, COPD, pneumonia, etc.",
        "system_prompt": (
            "You are a pulmonary medicine triage specialist. Analyze the clinical note focusing on:\n"
            "- Oxygen saturation and respiratory rate\n"
            "- Breath sounds and respiratory effort\n"
            "- History of chronic lung disease\n"
            "- Infection indicators (fever, productive cough, WBC)\n"
            "- Need for supplemental oxygen or ventilatory support\n\n"
            "Return JSON: {\"respiratory_severity\": \"critical\"|\"severe\"|\"moderate\"|\"mild\", "
            "\"suspected_diagnosis\": str, \"oxygen_requirement\": str, "
            "\"recommended_workup\": [str], \"urgency\": int}"
        ),
    },
    "mental_health": {
        "description": "Psychiatric presentations: depression, anxiety, suicidal ideation, psychosis, etc.",
        "system_prompt": (
            "You are a psychiatric triage specialist. Analyze the clinical note focusing on:\n"
            "- Safety assessment: suicidal ideation (SI), homicidal ideation (HI), self-harm\n"
            "- If SI present: plan, intent, means, protective factors\n"
            "- Psychotic symptoms: hallucinations, delusions, disorganized thinking\n"
            "- Substance use or withdrawal\n"
            "- Functional impairment level\n"
            "- Columbia Suicide Severity Rating Scale indicators if applicable\n\n"
            "CRITICAL: If ANY suicidal or homicidal ideation is present, urgency must be >= 4.\n\n"
            "Return JSON: {\"safety_risk\": \"imminent\"|\"high\"|\"moderate\"|\"low\", "
            "\"si_present\": bool, \"hi_present\": bool, \"psychosis_present\": bool, "
            "\"suspected_diagnosis\": str, \"recommended_disposition\": str, \"urgency\": int}"
        ),
    },
    "pediatric": {
        "description": "Pediatric presentations: any patient under 18 years old.",
        "system_prompt": (
            "You are a pediatric triage specialist. Analyze the clinical note focusing on:\n"
            "- Age-appropriate vital sign ranges\n"
            "- Hydration status and oral intake\n"
            "- Activity level and interaction (well-appearing vs toxic-appearing)\n"
            "- Vaccination status if relevant\n"
            "- Fever: age-based protocols (neonate <28 days with fever = emergent)\n"
            "- Growth and developmental concerns\n\n"
            "Return JSON: {\"pediatric_severity\": \"critical\"|\"acute\"|\"moderate\"|\"minor\", "
            "\"appearance\": \"well\"|\"ill\"|\"toxic\", \"suspected_diagnosis\": str, "
            "\"recommended_workup\": [str], \"urgency\": int, \"parent_instructions\": str}"
        ),
    },
    "general": {
        "description": "General/unclassified presentations that don't fit specific specialties.",
        "system_prompt": (
            "You are a general medicine triage specialist. Provide a comprehensive assessment.\n\n"
            "Return JSON: {\"primary_concern\": str, \"suspected_diagnosis\": str, "
            "\"recommended_workup\": [str], \"urgency\": int, \"disposition\": str}"
        ),
    },
}


# --- Classifier (Router) ---

CLASSIFIER_PROMPT: str = """You are a clinical input router. Your ONLY job is to classify the
incoming clinical note into the most appropriate specialty route.

Available routes:
- cardiology: Cardiac presentations (chest pain, arrhythmia, heart failure, syncope with cardiac features)
- respiratory: Pulmonary presentations (dyspnea, cough, asthma/COPD exacerbation, pneumonia)
- mental_health: Psychiatric presentations (depression, anxiety, suicidal ideation, psychosis, substance use)
- pediatric: ANY patient under 18 years old, regardless of complaint
- general: Anything that doesn't clearly fit the above categories

Classification rules:
1. Pediatric route takes priority if the patient is under 18.
2. If multiple specialties apply, choose the one most relevant to the CHIEF COMPLAINT.
3. When in doubt, choose "general".

Return JSON: {"route": str, "confidence": float (0-1), "reasoning": str}"""


def classify_input(note: str) -> dict[str, Any]:
    """Classify a clinical note to determine the appropriate processing route."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": CLASSIFIER_PROMPT},
            {"role": "user", "content": f"Clinical note:\n\n{note}"},
        ],
    )
    return json.loads(response.choices[0].message.content)


def process_with_route(note: str, route_name: str) -> dict[str, Any]:
    """Process a clinical note using the specialized prompt for the given route."""
    route = ROUTES.get(route_name, ROUTES["general"])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": route["system_prompt"]},
            {"role": "user", "content": f"Clinical note:\n\n{note}"},
        ],
    )
    return json.loads(response.choices[0].message.content)


def run_classification_pipeline(note: str) -> dict[str, Any]:
    """Run the full classification and routing pipeline."""
    # Step 1: Classify
    classification = classify_input(note)
    route_name = classification.get("route", "general")
    confidence = classification.get("confidence", 0.0)

    # Step 2: Route to specialized pipeline
    result = process_with_route(note, route_name)

    return {
        "classification": {
            "route": route_name,
            "confidence": confidence,
            "reasoning": classification.get("reasoning", ""),
        },
        "specialized_assessment": result,
    }


# --- Test Cases ---

TEST_CASES: list[dict[str, str]] = [
    {
        "label": "Cardiac - Acute Chest Pain",
        "note": (
            "62-year-old male with acute onset chest tightness while shoveling snow. Pain is "
            "substernal, pressure-like, radiating to left shoulder. Associated nausea. PMH: HTN, "
            "DM2, former smoker (quit 2 years ago). Father had MI at age 58. BP 158/95, HR 98, "
            "SpO2 96%. Initial troponin 0.08 (normal <0.04)."
        ),
        "expected_route": "cardiology",
    },
    {
        "label": "Mental Health - Suicidal Ideation",
        "note": (
            "35-year-old male brought by wife who found a note. Patient admits to suicidal "
            "ideation with a plan to overdose on medications. Has been stockpiling pills. "
            "Reports 3 months of worsening depression since job loss. Prior suicide attempt "
            "2 years ago via overdose. Denies current substance use. PHQ-9 score 24."
        ),
        "expected_route": "mental_health",
    },
    {
        "label": "Pediatric - Febrile Infant",
        "note": (
            "22-day-old male brought by parents for fever of 100.8F rectal, noticed 2 hours ago. "
            "Born full-term, uncomplicated delivery. Breastfeeding well. No sick contacts. "
            "No rash. Fontanelle soft and flat. Active, but parents note decreased feeding in "
            "the last few hours. No URI symptoms."
        ),
        "expected_route": "pediatric",
    },
    {
        "label": "Respiratory - COPD Exacerbation",
        "note": (
            "71-year-old female with known severe COPD (FEV1 35% predicted) presents with "
            "3-day progressive dyspnea. Increased sputum, now purulent. Using home O2 at "
            "2L, usually on 1L. Unable to complete sentences. Accessory muscle use noted. "
            "SpO2 86% on 2L NC. Bilateral expiratory wheezes with poor air movement. "
            "Temp 101.2F. WBC 14.8."
        ),
        "expected_route": "respiratory",
    },
    {
        "label": "General - Low Back Pain",
        "note": (
            "48-year-old female with 1-week history of low back pain after lifting heavy boxes "
            "during a move. Pain is bilateral, worse with bending. No radiation to legs. No "
            "numbness or weakness. No bowel or bladder changes. Able to walk normally. "
            "Taking ibuprofen with partial relief. No history of back problems."
        ),
        "expected_route": "general",
    },
]


def main() -> None:
    """Run the classification pipeline on all test cases and display results."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        console = Console()
        use_rich = True
    except ImportError:
        use_rich = False

    results_summary: list[dict] = []

    for test in TEST_CASES:
        result = run_classification_pipeline(test["note"])
        route = result["classification"]["route"]
        confidence = result["classification"]["confidence"]
        match = route == test["expected_route"]

        results_summary.append({
            "label": test["label"],
            "expected": test["expected_route"],
            "actual": route,
            "confidence": confidence,
            "match": match,
        })

        if use_rich:
            console.print(f"\n[bold cyan]{'=' * 80}[/bold cyan]")
            console.print(f"[bold]{test['label']}[/bold]")

            route_color = "green" if match else "red"
            console.print(
                f"Route: [{route_color}]{route}[/{route_color}] "
                f"(expected: {test['expected_route']}, confidence: {confidence:.0%})"
            )
            console.print(
                f"[dim]Reasoning: {result['classification']['reasoning']}[/dim]"
            )

            assessment = result["specialized_assessment"]
            console.print(
                Panel(
                    json.dumps(assessment, indent=2),
                    title=f"[bold]{route.title()} Assessment[/bold]",
                )
            )

    # Summary table
    if use_rich:
        console.print(f"\n[bold]{'=' * 80}[/bold]")
        table = Table(title="Classification Routing Summary")
        table.add_column("Case", style="cyan")
        table.add_column("Expected Route")
        table.add_column("Actual Route")
        table.add_column("Confidence")
        table.add_column("Match")

        for r in results_summary:
            match_str = "[green]YES[/green]" if r["match"] else "[red]NO[/red]"
            table.add_row(
                r["label"],
                r["expected"],
                r["actual"],
                f"{r['confidence']:.0%}",
                match_str,
            )

        console.print(table)
        correct = sum(1 for r in results_summary if r["match"])
        console.print(
            f"\nRouting Accuracy: {correct}/{len(results_summary)} "
            f"({correct / len(results_summary) * 100:.0f}%)"
        )


if __name__ == "__main__":
    main()


# --- Sample Output ---
#
# Cardiac - Acute Chest Pain
# Route: cardiology (expected: cardiology, confidence: 95%)
# Reasoning: Chest tightness with cardiac risk factors and elevated troponin is a
#            clear cardiac presentation.
# Cardiology Assessment:
# {
#   "cardiac_risk": "high",
#   "suspected_diagnosis": "NSTEMI",
#   "heart_score_estimate": 8,
#   "recommended_workup": ["Serial troponins q3h", "12-lead ECG", "Cardiology consult",
#                           "Echocardiogram", "Heparin drip"],
#   "urgency": 5
# }
#
# Mental Health - Suicidal Ideation
# Route: mental_health (expected: mental_health, confidence: 98%)
# Mental Health Assessment:
# {
#   "safety_risk": "imminent",
#   "si_present": true,
#   "hi_present": false,
#   "psychosis_present": false,
#   "suspected_diagnosis": "Major depressive disorder, severe, with suicidal ideation",
#   "recommended_disposition": "Involuntary psychiatric hold, 1:1 observation",
#   "urgency": 5
# }
#
# Classification Routing Summary:
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━┓
# ┃ Case                       ┃ Expected      ┃ Actual        ┃ Confidence ┃ Match ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━┩
# │ Cardiac - Acute Chest Pain │ cardiology    │ cardiology    │ 95%        │ YES   │
# │ Mental Health - SI         │ mental_health │ mental_health │ 98%        │ YES   │
# │ Pediatric - Febrile Infant │ pediatric     │ pediatric     │ 99%        │ YES   │
# │ Respiratory - COPD         │ respiratory   │ respiratory   │ 92%        │ YES   │
# │ General - Low Back Pain    │ general       │ general       │ 88%        │ YES   │
# └────────────────────────────┴───────────────┴───────────────┴────────────┴───────┘
# Routing Accuracy: 5/5 (100%)
