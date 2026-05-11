import os
import json
import re
from dotenv import load_dotenv
from openai import OpenAI

class PairwiseVerifier:
    def __init__(self, model: str = "meta-llama/llama-3.3-70b-instruct", api_key_env: str = "OPENROUTER_API_KEY"):
        load_dotenv()
        self.client = OpenAI(
            api_key=os.getenv(api_key_env),
            # GitHub's official marketplace endpoint
            base_url="https://openrouter.ai/api/v1" 
        )
        self.model = model

    def _clean_json(self, content: str) -> str:
        content = content.strip()
        # Removes markdown code fences
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
        return content.strip()

    def _build_prompt(self, question, opt1_letter, opt1_text, opt2_letter, opt2_text) -> str:
        return f"""
You are a strict medical reasoning verifier.

Compare the two answer options carefully.

Rules:
- Evaluate ONLY the two provided options.
- Use medically correct reasoning.
- Exactly one option is correct.
- Return ONLY valid JSON.
- Do not use markdown.
- Do not add extra commentary.

Question:
{question}

Option {opt1_letter}: {opt1_text}

Option {opt2_letter}: {opt2_text}

Return this schema exactly:

{{
  "reasoning_trace": [
    "step 1",
    "step 2",
    "step 3"
  ],
  "selected_letter": "A",
  "justification": "short justification"
}}
"""

    def verify_pair(self, question, opt1_letter, opt1_text, opt2_letter, opt2_text) -> dict:
        prompt = self._build_prompt(
            question,
            opt1_letter,
            opt1_text,
            opt2_letter,
            opt2_text
        )

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a JSON generator. "
                        "Return ONLY valid JSON. "
                        "Do not use markdown. "
                        "Do not add explanations."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        content = response.choices[0].message.content.strip()

        print("\nRAW MODEL OUTPUT:\n")
        print(content)

        cleaned_content = self._clean_json(content)

        try:
            return json.loads(cleaned_content)

        except json.JSONDecodeError as e:
            print("\nFAILED JSON:\n")
            print(cleaned_content)
            raise e