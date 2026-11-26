"""Three-class Turkish sentiment via the System One Choice API."""

import os
import re
import time

import httpx


DEFAULT_JEV_MODEL = "jev-1.14.0"
TYPESAFE_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
OPENCODE_FREE_MODEL = "jev-1.13-free"
JEV_ENDPOINTS = {
    "typesafe": TYPESAFE_ENDPOINT,
    "opencode": "https://opencode.ai/inference/systemone/v1/systemone",
}
SENTIMENT_QUESTION = {
    "type": "choice",
    "instructions": "Türkçe kullanıcı yorumunun genel duygu yönü nedir?",
    "criteria": {
        "positive": "Genel olarak memnuniyet, övgü veya olumlu değerlendirme ifade eder.",
        "negative": "Genel olarak memnuniyetsizlik, eleştiri veya olumsuz değerlendirme ifade eder.",
        "neutral": "Genel olarak belirgin bir olumlu veya olumsuz değerlendirme içermez ya da iki yön birbirini dengeler.",
    },
}


def resolve_jev_model(provider, model=None):
    if provider == "opencode":
        if model not in (None, OPENCODE_FREE_MODEL):
            raise ValueError(f"OpenCode Jev only supports {OPENCODE_FREE_MODEL} in this runner.")
        return OPENCODE_FREE_MODEL
    if provider != "typesafe":
        raise ValueError(f"Unknown Jev provider: {provider}")
    model = DEFAULT_JEV_MODEL if model is None else model
    if not re.fullmatch(r"jev-\d+\.\d+\.\d+", model):
        raise ValueError("Jev requires a pinned model version, such as jev-1.14.0.")
    return model


class JevZeroShotSentimentClassifier:
    """Send one review as state and keep the returned decision and diagnostics."""

    def __init__(self, model=None, request_timeout=30.0, provider="typesafe"):
        self.model = resolve_jev_model(provider, model)
        self.provider = provider
        self.endpoint = JEV_ENDPOINTS[provider]
        api_key_env = "OPENCODE_API_KEY" if provider == "opencode" else "TYPESAFE_API_KEY"
        api_key = os.getenv(api_key_env, "").strip()
        if provider == "typesafe" and not api_key:
            raise ValueError("Set TYPESAFE_API_KEY before running Jev.")
        self.client = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            timeout=request_timeout,
        )

    def classify(self, text):
        payload = {
            "model": self.model,
            "state": text,
            "questions": {"sentiment": SENTIMENT_QUESTION},
        }
        result = dict(prediction=None, status="failed", model=self.model, attempt=1)
        try:
            response = self.client.post(self.endpoint, json=payload)
            if response.status_code in {429, 529}:
                retry_after = response.headers.get("Retry-After", "")
                delay = int(retry_after) if re.fullmatch(r"[0-9]{1,9}", retry_after) else 2
                if delay <= 30:
                    time.sleep(max(1, delay))
                    result["attempt"] += 1
                    response = self.client.post(self.endpoint, json=payload)
        except httpx.TransportError as error:
            return dict(result, error=type(error).__name__, stop_run=True)

        if not response.is_success:
            code = response.status_code
            provider_name = "OpenCode" if self.provider == "opencode" else "TypeSafe"
            return dict(result, error=f"{provider_name} HTTP {code}", error_code=f"http_{code}",
                        stop_run=code in {401, 402, 403, 404, 429} or code >= 500)

        try:
            body = response.json()
            usage = body["usage"]
            token_counts = (usage["input_tokens"], usage["output_tokens"])
            if any(type(count) is not int or count < 0 for count in token_counts):
                raise ValueError("Invalid token usage.")
            result.update(prompt_tokens=token_counts[0], output_tokens=token_counts[1])
            if body["model"] != self.model:
                return dict(result, error="Provider returned a different Jev model.", stop_run=True)
            if self.provider == "opencode":
                cost = usage.get("cost")
                billing = usage.get("opencode")
                if (type(cost) not in (int, float) or cost != 0
                        or not isinstance(billing, dict) or billing.get("billing_source") != "free"):
                    return dict(result, error="OpenCode did not confirm free usage.", stop_run=True)
                result.update(cost=cost, billing_source=billing["billing_source"])
            answer = body["answers"]["sentiment"]
            choice = answer["choice"]
            probabilities = answer["probabilities"]
            confidence = answer["confidence"]
            labels = SENTIMENT_QUESTION["criteria"]
            if (answer["type"] != "choice" or choice not in labels
                    or not isinstance(probabilities, dict) or probabilities.keys() != labels.keys()
                    or any(type(value) not in (int, float) or not 0 <= value <= 1
                           for value in (*probabilities.values(), confidence))):
                raise ValueError("Invalid sentiment choice.")
        except (KeyError, TypeError, ValueError):
            return dict(result, error="Invalid Jev sentiment response.", error_code="invalid_response")
        return dict(result, status="success", prediction=choice, error=None,
                    probabilities=probabilities, confidence=confidence)
