"""LLM reasoning layer with Groq primary and Ollama fallback."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List

import requests


@dataclass
class LLMReasonerConfig:
    """Configuration for LLM provider routing."""

    # Use default_factory so env vars are read at instance creation time,
    # after load_dotenv() has already run in app/inference entrypoints.
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"))
    groq_url: str = field(
        default_factory=lambda: os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1/chat/completions")
    )
    ollama_url: str = field(default_factory=lambda: os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.1:latest"))
    timeout_sec: int = field(default_factory=lambda: int(os.getenv("LLM_TIMEOUT_SEC", "30")))


class LLMReasoner:
    """Generate fraud/uncertainty rationale from structured pipeline signals."""

    def __init__(self, config: LLMReasonerConfig | None = None) -> None:
        self.config = config or LLMReasonerConfig()

    @staticmethod
    def _prompt(payload: Dict[str, object]) -> str:
        """Build constrained prompt requiring JSON output."""
        return (
            "You are an insurance-claim AI reasoning assistant.\n"
            "Given structured evidence, produce strict JSON with keys:\n"
            "summary, final_verdict, confidence_band, uncertainty_explanation, decision, "
            "recommended_actions (array of 2-4 strings), risk_factors (array), positive_factors (array).\n"
            "Decision must align with risk: LOW->APPROVE, MEDIUM->REVIEW, HIGH->REJECT, UNKNOWN->REVIEW.\n"
            "Do not contradict provided fraud_risk/final_case_risk in verdict text.\n"
            "Keep it concise and factual. Do not invent evidence.\n\n"
            f"EVIDENCE_JSON:\n{json.dumps(payload, ensure_ascii=True)}"
        )

    @staticmethod
    def _safe_parse_json(text: str) -> Dict[str, object]:
        """Parse JSON text robustly with fallback structure."""
        try:
            return json.loads(text)
        except Exception:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except Exception:
                    pass
        return {}

    def _call_groq(self, prompt: str) -> Dict[str, object]:
        """Call Groq OpenAI-compatible chat completion endpoint."""
        if not self.config.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not set.")
        headers = {
            "Authorization": f"Bearer {self.config.groq_api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.config.groq_model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": "Return JSON only."},
                {"role": "user", "content": prompt},
            ],
        }
        resp = requests.post(self.config.groq_url, headers=headers, json=body, timeout=self.config.timeout_sec)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        parsed = self._safe_parse_json(content)
        if not parsed:
            raise RuntimeError("Groq returned non-JSON response.")
        parsed["provider"] = "groq"
        parsed["model"] = self.config.groq_model
        return parsed

    def _call_ollama(self, prompt: str) -> Dict[str, object]:
        """Call local Ollama model as fallback."""
        body = {
            "model": self.config.ollama_model,
            "prompt": f"Return JSON only.\n{prompt}",
            "stream": False,
            "options": {"temperature": 0.2},
        }
        resp = requests.post(self.config.ollama_url, json=body, timeout=self.config.timeout_sec)
        resp.raise_for_status()
        data = resp.json()
        content = str(data.get("response", "")).strip()
        parsed = self._safe_parse_json(content)
        if not parsed:
            raise RuntimeError("Ollama returned non-JSON response.")
        parsed["provider"] = "ollama"
        parsed["model"] = self.config.ollama_model
        return parsed

    @staticmethod
    def _deterministic_fallback(payload: Dict[str, object], errors: List[str]) -> Dict[str, object]:
        """Deterministic reasoning when LLMs are unavailable."""
        decision_conf = float(payload.get("decision_confidence", 0.0))
        fraud_risk = str(payload.get("fraud_risk", payload.get("final_case_risk", "UNKNOWN")))
        mismatch_reasons = payload.get("mismatch_reasons", [])
        uncertainty_factors = payload.get("uncertainty_factors", [])
        confidence_band = "HIGH" if decision_conf >= 0.75 else "MEDIUM" if decision_conf >= 0.5 else "LOW"
        decision_map = {"LOW": "APPROVE", "MEDIUM": "REVIEW", "HIGH": "REJECT", "UNKNOWN": "REVIEW"}
        return {
            "summary": "Rule-based reasoning fallback used due to unavailable LLM providers.",
            "final_verdict": f"Fraud risk is {fraud_risk}.",
            "confidence_band": confidence_band,
            "uncertainty_explanation": ", ".join(uncertainty_factors) if uncertainty_factors else "No major uncertainty factors.",
            "decision": decision_map.get(fraud_risk, "REVIEW"),
            "recommended_actions": [
                "Review annotated detections manually",
                "Verify claim location against visual evidence",
                "Escalate if mismatch persists across multiple images",
            ],
            "risk_factors": list(mismatch_reasons)[:4],
            "positive_factors": [],
            "provider": "rule_fallback",
            "model": "none",
            "provider_errors": errors,
        }

    @staticmethod
    def _expected_decision(risk: str) -> str:
        """Return expected decision token for a risk level."""
        return {"LOW": "APPROVE", "MEDIUM": "REVIEW", "HIGH": "REJECT", "UNKNOWN": "REVIEW"}.get(risk, "REVIEW")

    def _enforce_consistency(self, payload: Dict[str, object], result: Dict[str, object]) -> Dict[str, object]:
        """Enforce that LLM output cannot contradict computed risk."""
        risk = str(payload.get("fraud_risk", payload.get("final_case_risk", "UNKNOWN"))).upper()
        total_detections = int(payload.get("total_detections", -1))
        expected_decision = self._expected_decision(risk)
        verdict = str(result.get("final_verdict", "")).strip()
        summary = str(result.get("summary", "")).strip()
        decision = str(result.get("decision", "")).strip().upper() or expected_decision

        contradiction = False
        if risk == "LOW" and ("reject" in verdict.lower() or "high risk" in verdict.lower()):
            contradiction = True
        if risk == "HIGH" and ("approve" in verdict.lower() or "low risk" in verdict.lower()):
            contradiction = True
        if risk == "MEDIUM" and ("approve" in verdict.lower() and "reject" in verdict.lower()):
            contradiction = True

        if contradiction:
            verdict = f"Fraud risk is {risk}."
            summary = "LLM output adjusted by consistency guardrail to match computed risk."

        # Hard safety rule: no detections means insufficient visual evidence.
        # Never auto-approve in this case, even if another module outputs LOW risk.
        if total_detections == 0:
            risk = "UNKNOWN"
            expected_decision = "REVIEW"
            verdict = "No visual damage was detected in the uploaded image; claim requires manual review."
            summary = "Insufficient visual evidence for automated approval."

        result["decision"] = expected_decision if decision not in {"APPROVE", "REVIEW", "REJECT"} else decision
        # Force decision to match computed risk for safety.
        result["decision"] = expected_decision
        result["final_verdict"] = verdict or f"Fraud risk is {risk}."
        result["summary"] = summary or "Reasoning generated from structured pipeline evidence."
        return result

    def reason(self, payload: Dict[str, object]) -> Dict[str, object]:
        """Generate reasoning using Groq first, then Ollama fallback."""
        prompt = self._prompt(payload)
        errors: List[str] = []
        try:
            return self._enforce_consistency(payload, self._call_groq(prompt))
        except Exception as exc:
            errors.append(f"groq: {exc}")
        try:
            result = self._call_ollama(prompt)
            result["provider_errors"] = errors
            return self._enforce_consistency(payload, result)
        except Exception as exc:
            errors.append(f"ollama: {exc}")
        return self._enforce_consistency(payload, self._deterministic_fallback(payload, errors))
