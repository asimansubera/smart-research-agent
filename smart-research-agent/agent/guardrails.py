"""
Guardrails — two layers of safety with no human-in-the-loop:

  1. input_guardrail  : blocks off-topic or harmful queries before the agent acts
  2. output_guardrail : ensures the agent's response is safe before it reaches the user

Both run in parallel with the main agent loop (fail-fast).
"""

from pydantic import BaseModel
from agents import (
    Agent,
    RunContextWrapper,
    Runner,
    input_guardrail,
    output_guardrail,
    GuardrailFunctionOutput,
)


# ── Pydantic schemas for structured guardrail outputs ─────────────────────────

class InputSafetyResult(BaseModel):
    is_safe: bool
    reason: str


class OutputSafetyResult(BaseModel):
    is_safe: bool
    reason: str


# ── Lightweight classifier agents ─────────────────────────────────────────────

_input_classifier = Agent(
    name="Input Safety Classifier",
    instructions="""
You are a content safety classifier for a research assistant.
Analyse the user's message and respond ONLY with a JSON object:
  { "is_safe": true/false, "reason": "brief explanation" }

Mark as UNSAFE (is_safe: false) if the message:
  - Asks for illegal information or instructions for harm
  - Contains abusive or hateful language
  - Is completely unrelated to research, facts, news, or calculations

Mark as SAFE (is_safe: true) for all legitimate research, learning,
news, maths, or factual questions.
""",
    output_type=InputSafetyResult,
)

_output_classifier = Agent(
    name="Output Safety Classifier",
    instructions="""
You are an output safety reviewer for a research assistant.
Review the agent's response and reply ONLY with a JSON object:
  { "is_safe": true/false, "reason": "brief explanation" }

Mark as UNSAFE (is_safe: false) if the response:
  - Contains step-by-step instructions for illegal activities
  - Includes hate speech or targeted harassment
  - Fabricates dangerous misinformation presented as fact

Mark as SAFE (is_safe: true) in all other cases.
""",
    output_type=OutputSafetyResult,
)


# ── Guardrail functions ───────────────────────────────────────────────────────

@input_guardrail
async def research_input_guardrail(
    ctx: RunContextWrapper,
    agent: Agent,
    user_input: str,
) -> GuardrailFunctionOutput:
    """Runs BEFORE the main agent sees the user's message."""
    result = await Runner.run(
        _input_classifier,
        user_input,
        context=ctx.context,
    )
    check: InputSafetyResult = result.final_output
    return GuardrailFunctionOutput(
        output_info=check,
        tripwire_triggered=not check.is_safe,
    )


@output_guardrail
async def research_output_guardrail(
    ctx: RunContextWrapper,
    agent: Agent,
    output: str,
) -> GuardrailFunctionOutput:
    """Runs AFTER the main agent produces a response, BEFORE it reaches the user."""
    result = await Runner.run(
        _output_classifier,
        output,
        context=ctx.context,
    )
    check: OutputSafetyResult = result.final_output
    return GuardrailFunctionOutput(
        output_info=check,
        tripwire_triggered=not check.is_safe,
    )
