# proposition_verifier.py

from dotenv import load_dotenv
from openai import OpenAI
import os
import json
import re


class PropositionVerifier:

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
        Removes markdown code fences from model output.
        """

        content = content.strip()

        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

        return content.strip()

    def _build_prompt(
        self,
        propositions: str,
        hypotheses: str
    ) -> str:

        return f"""
You are a strict proposition-based verifier.

You MUST evaluate hypotheses ONLY using the provided propositions.

Rules:
- Do NOT use external knowledge.
- Do NOT hallucinate missing facts.
- TRUE = directly supported or logically entailed by propositions
- FALSE = contradicted by propositions
- UNKNOWN = insufficient evidence

Important:
- Treat propositions as the complete world knowledge.
- Use careful logical reasoning.
- Do not assume synonyms unless explicitly stated.

Return ONLY valid JSON.

Propositions:
{propositions}

Hypotheses:
{hypotheses}

Output format:
{{
  "H01": {{
    "label": "TRUE/FALSE/UNKNOWN",
    "evidence": ["Pxxx", "Pyyy"]
  }}
}}
"""

    def verify(
        self,
        propositions: str,
        hypotheses: str
    ) -> dict:

        prompt = self._build_prompt(
            propositions=propositions,
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