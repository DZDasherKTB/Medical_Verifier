import os
import json
import re
from dotenv import load_dotenv
from openai import OpenAI

class TripletVerifier:
    def __init__(self, model: str = "meta-llama/llama-3.3-70b-instruct", api_key_env: str = "OPENROUTER_API_KEY"):
        load_dotenv()
        self.client = OpenAI(
            api_key=os.getenv(api_key_env),
            # GitHub's official marketplace endpoint or OpenRouter
            base_url="[https://openrouter.ai/api/v1](https://openrouter.ai/api/v1)" 
        )
        self.model = model

    def _clean_json(self, content: str) -> str:
        content = content.strip()
        # Removes markdown code fences using \x60 to represent backticks 
        # This prevents text editors from breaking the string during copy-paste
        content = re.sub(r"^\x60\x60\x60(?:json)?\s*", "", content)
        content = re.sub(r"\s*\x60\x60\x60$", "", content)
        return content.strip()

    def _build_prompt(self, question, opt1_letter, opt1_text, opt2_letter, opt2_text, opt3_letter, opt3_text) -> str:
        return f"""
You are a strict medical reasoning verifier.

Compare the three answer options carefully.

Rules:
- Evaluate ONLY the three provided options.
- Use medically correct reasoning.
- Exactly one option is correct.
- Return ONLY valid JSON.
- Do not use markdown.
- Do not add extra commentary.

Question:
{question}

Option {opt1_letter}: {opt1_text}

Option {opt2_letter}: {opt2_text}

Option {opt3_letter}: {opt3_text}

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

    def verify_triplet(self, question, opt1_letter, opt1_text, opt2_letter, opt2_text, opt3_letter, opt3_text) -> dict:
        prompt = self._build_prompt(
            question,
            opt1_letter, opt1_text,
            opt2_letter, opt2_text,
            opt3_letter, opt3_text
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
