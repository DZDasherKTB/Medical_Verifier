import os
import json
import re
from dotenv import load_dotenv
from openai import AzureOpenAI


class TripletVerifier:

    def __init__(
        self,
        deployment_name: str = "gpt-5.5",
        api_key_env: str = "AZURE_OPENAI_API_KEY",
        endpoint_env: str = "AZURE_OPENAI_ENDPOINT",
        api_version: str = "2024-12-01-preview"
    ):

        load_dotenv()

        self.api_key = os.getenv(api_key_env)
        self.endpoint = os.getenv(endpoint_env)

        if not self.api_key:
            raise ValueError(
                f"Missing API key in environment variable: {api_key_env}"
            )

        if not self.endpoint:
            raise ValueError(
                f"Missing Azure endpoint in environment variable: {endpoint_env}"
            )

        self.client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
        )

        # Deployment name in Azure
        self.model = deployment_name

    def _clean_json(self, content: str) -> str:

        content = content.strip()

        # Remove markdown fences if model accidentally adds them
        content = re.sub(r"^\x60\x60\x60(?:json)?\s*", "", content)
        content = re.sub(r"\s*\x60\x60\x60$", "", content)

        return content.strip()

    def _build_prompt(
        self,
        question,
        opt1_letter,
        opt1_text,
        opt2_letter,
        opt2_text,
        opt3_letter,
        opt3_text
    ) -> str:

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

    def verify_triplet(
        self,
        question,
        opt1_letter,
        opt1_text,
        opt2_letter,
        opt2_text,
        opt3_letter,
        opt3_text
    ) -> dict:

        prompt = self._build_prompt(
            question,
            opt1_letter,
            opt1_text,
            opt2_letter,
            opt2_text,
            opt3_letter,
            opt3_text
        )

        try:

            response = self.client.chat.completions.create(
                model=self.model,
                max_completion_tokens=2048,
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

        except Exception as e:

            import traceback

            print("\nFULL ERROR TRACEBACK:\n")
            traceback.print_exc()

            print(f"\nERROR: {str(e)}\n")

            raise e