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
        """
        Removes markdown code fences if the model outputs them.
        """

        content = content.strip()

        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

        return content.strip()

    def _build_prompt(
        self,
        question: str,
        options: list
    ) -> str:

        options_text = "\n".join(
            [
                f"{chr(65 + i)}. {option}"
                for i, option in enumerate(options)
            ]
        )

        return f"""
You are a strict medical reasoning system.

Your task is to solve the medical multiple choice question using careful reasoning.

Rules:
- Use only medically valid reasoning.
- Analyze every option independently.
- Do not skip reasoning steps.
- YES = option is correct
- NO = option is incorrect
- Exactly one option should normally be YES unless the question explicitly allows multiple answers.
- Be logically consistent.

You must:
1. Read the question carefully
2. Generate a detailed reasoning trace
3. Pick the final answer
4. Evaluate every option with YES or NO
5. Give a short reason for every option

Return ONLY valid JSON.

Question:
{question}

Options:
{options_text}

Output format:
{{
  "reasoning_trace": [
    "step 1",
    "step 2"
  ],

  "selected_answer": "A",

  "selected_option_text": "full option text",

  "options_analysis": {{
    "A": {{
      "option": "full option text",
      "label": "YES",
      "reason": "why correct"
    }},
    "B": {{
      "option": "full option text",
      "label": "NO",
      "reason": "why incorrect"
    }},
    "C": {{
      "option": "full option text",
      "label": "NO",
      "reason": "why incorrect"
    }},
    "D": {{
      "option": "full option text",
      "label": "NO",
      "reason": "why incorrect"
    }}
  }}
}}
"""

    def solve_mcq(
        self,
        question: str,
        options: list
    ) -> dict:

        prompt = self._build_prompt(
            question=question,
            options=options
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

    options = [
        "IgG",
        "IgM",
        "IgA",
        "IgE"
    ]

    result = verifier.solve_mcq(
        question=question,
        options=options
    )

    print(json.dumps(result, indent=2))