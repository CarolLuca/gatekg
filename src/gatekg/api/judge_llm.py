from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from typing import Any

from google.genai import types

from gatekg.contracts import JudgeResult
from gatekg.utils.config import Config, load_config
from gatekg.utils.llm_concurrency import LLM_REQUEST_LIMITER
from gatekg.utils.prompt_truncation import fit_judge_context
from gatekg.utils.retry import call_with_retries

from .google_client import build_google_genai_client

config: Config = load_config()

MAX_RETRIES = config.llm_runtime.max_attempts
RETRY_DELAY_SECONDS = config.llm_runtime.retry_delay_seconds
JUDGE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 1,
        },
        "explanation": {
            "type": "string",
        },
    },
    "required": ["score", "explanation"],
}


def _serialize_context(context: str | list[str] | None) -> str:
    if context is None:
        return ""
    if isinstance(context, str):
        return context
    return "\n\n".join(context)


def _judge_prompt(
    *,
    question: str,
    gold: str,
    prediction: str,
    context: str | list[str] | None,
) -> str:
    context_block = _serialize_context(context)
    if context_block:
        context_section = f"Context:\n{context_block}\n\n"
    else:
        context_section = ""

    return (
        "You are grading the correctness of an answer.\n"
        "Return only JSON with keys `score` and `explanation`.\n"
        "The score must be either 1 or 0.\n"
        "Set `score` to 1 only when the prediction is fully correct with respect to the gold answer.\n"
        "Set `score` to 0 when the prediction is partially correct, incomplete, ambiguous, unsupported, contradictory, or wrong.\n"
        "Treat paraphrases and equivalent formatting as correct when they preserve the full meaning of the gold answer.\n"
        "Use the context only as supporting evidence when it is provided.\n\n"
        f"Question:\n{question}\n\n"
        f"Gold answer:\n{gold}\n\n"
        f"Prediction:\n{prediction}\n\n"
        f"{context_section}"
        "Scoring rubric:\n"
        "- 1: fully correct\n"
        "- 0: not fully correct\n"
    )


def _normalize_score(raw_score: Any) -> int:
    if isinstance(raw_score, bool):
        return int(raw_score)

    if isinstance(raw_score, (int, float)):
        numeric_score = float(raw_score)
    else:
        normalized_text = str(raw_score).strip().lower()
        if normalized_text in {"correct", "true", "yes"}:
            return 1
        if normalized_text in {"incorrect", "false", "no"}:
            return 0
        numeric_score = float(normalized_text)

    if numeric_score == 0:
        return 0
    if numeric_score == 1:
        return 1
    raise ValueError(f"Judge score must be 0 or 1, got {raw_score!r}")


class JudgeLLM:
    def __init__(self):
        self.model_name = config.google_judge_model_name
        self.client = build_google_genai_client(
            config=config,
            timeout_seconds=config.llm_runtime.timeout_seconds,
        )

    def _config(self) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            temperature=0.0,
            top_p=1.0,
            candidate_count=1,
            seed=0,
            response_mime_type="application/json",
            response_json_schema=JUDGE_RESPONSE_SCHEMA,
        )

    def _parse_response(self, response_text: str) -> JudgeResult:
        payload = json.loads(response_text)
        raw_score = payload["score"]
        explanation = str(payload["explanation"]).strip()
        score = _normalize_score(raw_score)
        if not explanation:
            explanation = "No explanation provided."
        return JudgeResult(score=score, explanation=explanation)

    def unsafe_judge(
        self,
        question: str,
        gold: str,
        prediction: str,
        context: str | list[str] | None = None,
    ) -> JudgeResult:
        with LLM_REQUEST_LIMITER.sync_slot():
            prompt, omitted = fit_judge_context(
                client=self.client,
                model=self.model_name,
                limit=config.input_token_cap_for("google"),
                context=_serialize_context(context),
                render=lambda text: _judge_prompt(
                    question=question, gold=gold, prediction=prediction, context=text
                ),
            )
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=self._config(),
            )
        if response.text is None:
            raise ValueError("Judge returned an empty response.")
        return replace(self._parse_response(response.text), context_chars_omitted=omitted)

    def judge(
        self,
        question: str,
        gold: str,
        prediction: str,
        context: str | list[str] | None = None,
    ) -> JudgeResult:
        return call_with_retries(
            lambda: self.unsafe_judge(question, gold, prediction, context),
            attempts=MAX_RETRIES,
            delay_seconds=RETRY_DELAY_SECONDS,
        )

    async def ajudge(
        self,
        question: str,
        gold: str,
        prediction: str,
        context: str | list[str] | None = None,
    ) -> JudgeResult:
        return await asyncio.to_thread(
            self.judge,
            question,
            gold,
            prediction,
            context,
        )
