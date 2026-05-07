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
        api_key_env: str = "GROQ_API_KEY_2"
    ):

        load_dotenv()

        self.client = OpenAI(
            api_key=os.getenv(api_key_env),
            base_url="https://api.groq.com/openai/v1"
        )

        self.model = model

    def _clean_json(self, content: str) -> str:

        content = content.strip()

        content = re.sub(
            r"^```(?:json)?\s*",
            "",
            content
        )

        content = re.sub(
            r"\s*```$",
            "",
            content
        )

        return content.strip()

    def _build_prompt(
        self,
        question: str,
        option_letter: str,
        option_text: str
    ) -> str:

        return f"""
You are a strict medical reasoning verifier.

Your task is to determine whether the given option correctly answers the medical question.

Rules:
- Evaluate ONLY the provided option.
- Use medically valid reasoning.
- YES = option is correct
- NO = option is incorrect
- Generate step-by-step reasoning.
- Be logically rigorous.

Return ONLY valid JSON.

Question:
{question}

Option:
{option_letter}. {option_text}

Output format:
{{
  "option_letter": "{option_letter}",

  "option_text": "{option_text}",

  "label": "YES or NO",

  "reasoning_trace": [
    "step 1",
    "step 2"
  ],

  "reason": "short final justification"
}}
"""

    def verify_option(
        self,
        question: str,
        option_letter: str,
        option_text: str
    ) -> dict:

        prompt = self._build_prompt(
            question=question,
            option_letter=option_letter,
            option_text=option_text
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


if __name__ == "__main__":

    verifier = ReasoningVerifier()

    question = (
        "Which immunoglobulin is the first antibody produced during "
        "primary immune response?"
    )

    options = {
        "A": "IgG",
        "B": "IgM",
        "C": "IgA",
        "D": "IgE"
    }

    final_results = {}

    for letter, text in options.items():

        result = verifier.verify_option(
            question=question,
            option_letter=letter,
            option_text=text
        )

        final_results[letter] = result

    print(
        json.dumps(
            final_results,
            indent=2
        )
    )