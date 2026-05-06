# reasoning_verifier.py

from dotenv import load_dotenv
from openai import OpenAI
import os
import json
import re


class ReasoningVerifier:

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        api_key_env: str = "GROQ_API_KEY"
    ):
        load_dotenv()

        self.client = OpenAI(
            api_key=os.getenv(api_key_env),
            base_url="https://api.groq.com/openai/v1"
        )

        self.model = model

    def _clean_json(self, content: str) -> str:
        """
        Removes markdown code fences if the model outputs them.
        """

        content = content.strip()

        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

        return content.strip()

    def _build_prompt(
        self,
        reasoning_trace: str,
        hypotheses: list
    ) -> str:

        hypotheses_text = "\n".join(hypotheses)

        return f"""
You are a strict reasoning verifier.

You MUST evaluate the hypotheses ONLY using the provided reasoning trace.

Rules:
- Do NOT use external knowledge.
- Do NOT hallucinate missing facts.
- TRUE = directly supported or logically inferable
- FALSE = contradicted by the reasoning trace
- UNKNOWN = insufficient information

Important:
- Treat the reasoning trace as the complete world knowledge.
- Use careful logical reasoning.
- Do not assume synonyms unless explicitly stated.

Return ONLY valid JSON.

Reasoning Trace:
{reasoning_trace}

Hypotheses:
{hypotheses_text}

Output format:
{{
  "H01": {{
    "label": "TRUE/FALSE/UNKNOWN",
    "evidence": ["quoted supporting sentence"]
  }}
}}
"""

    def verify(
        self,
        reasoning_trace: str,
        hypotheses: list
    ) -> dict:

        prompt = self._build_prompt(
            reasoning_trace=reasoning_trace,
            hypotheses=hypotheses
        )

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        content = response.choices[0].message.content

        cleaned_content = self._clean_json(content)

        result = json.loads(cleaned_content)

        return result