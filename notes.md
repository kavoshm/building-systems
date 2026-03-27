# Building Multi-Step LLM Systems — Study Notes

## Architecture Patterns

### 1. Sequential Chain (Pipeline)

```
Input -> [Step 1: Classify] -> [Step 2: Extract] -> [Step 3: Reason] -> [Step 4: Format] -> Output
```

Each step is a separate LLM call with its own system prompt optimized for that specific task.
The output of each step is validated before passing to the next.

**Healthcare example:** Clinical note arrives -> classify specialty -> extract relevant findings
-> assess urgency -> generate structured output with ICD-10 codes.

### 2. Router Pattern

```
Input -> [Classifier] -+-> Route A: Cardiology Pipeline
                       +-> Route B: Mental Health Pipeline
                       +-> Route C: Pediatric Pipeline
                       +-> Route D: General Pipeline
```

The classifier determines which specialized pipeline handles the input. Each route has prompts
optimized for that clinical domain.

**Why this matters:** A prompt optimized for cardiac presentations will outperform a generic
prompt when classifying chest pain. Domain-specific routing improves accuracy.

### 3. Guardrail Sandwich

```
Input -> [Input Moderation] -> [Core Processing] -> [Output Moderation] -> Output
```

Both input and output pass through safety checks. In healthcare, this prevents:
- Input: Processing non-clinical or adversarial inputs
- Output: Generating specific treatment plans, prescriptions, or definitive diagnoses
  (which could create liability)

### 4. Evaluation / Self-Check

```
Input -> [Process] -> Output -> [Evaluate Output] -> Final Output or Retry
```

The system checks its own output for quality, consistency, and safety before returning it.
If the evaluation fails, the system can retry with additional context.

---

## Implementation Patterns

### Chaining with Context Accumulation

Each step adds to a growing context object that flows through the pipeline:

```python
context = {"raw_input": note_text}
context["classification"] = classify(note_text)           # Step 1
context["extraction"] = extract(note_text, context)        # Step 2 uses Step 1 output
context["assessment"] = assess(context)                    # Step 3 uses all prior context
context["output"] = format_output(context)                 # Step 4 formats everything
```

### Error Handling Between Steps

Each step should:
1. Validate its input (did the previous step succeed?)
2. Have a timeout
3. Have retry logic (up to N attempts)
4. Have a fallback (graceful degradation if a step fails)

```python
def safe_step(step_fn, input_data, max_retries=3):
    for attempt in range(max_retries):
        try:
            result = step_fn(input_data)
            if validate(result):
                return result
        except Exception as e:
            log(f"Step failed attempt {attempt}: {e}")
    return fallback_result()
```

### Model Selection by Step

Not every step needs the most capable model:

| Step | Model | Reasoning |
|------|-------|-----------|
| Input classification | gpt-4o-mini | Simple routing, fast |
| Moderation | gpt-4o-mini | Pattern matching, fast |
| Clinical extraction | gpt-4o | Needs accuracy |
| Reasoning/assessment | gpt-4o | Complex task |
| Output formatting | gpt-4o-mini | Template filling |

This optimizes cost and latency while maintaining accuracy where it matters.

---

## Key Insights for Healthcare Systems

1. **Decomposition is debuggability.** When a clinical classification is wrong, a chained system
   lets you see exactly which step failed — was it the extraction, the reasoning, or the
   formatting? A monolithic prompt gives you no visibility.

2. **Routing enables specialization.** A cardiologist doesn't use the same checklist as a
   psychiatrist. Your LLM shouldn't either. Route to specialized prompts.

3. **Moderation is non-negotiable.** In healthcare, an unmoderated LLM could generate harmful
   advice. Always wrap core processing in safety layers.

4. **Intermediate outputs are audit trails.** In regulated healthcare environments, you need to
   show how the system arrived at a classification. Chained steps with logged intermediates
   provide this naturally.

5. **Cost control through architecture.** A 5-step pipeline using gpt-4o-mini for 3 steps and
   gpt-4o for 2 steps costs significantly less than running everything through gpt-4o, with
   minimal accuracy loss.
