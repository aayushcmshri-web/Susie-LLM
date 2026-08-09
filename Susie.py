import os
from pathlib import Path

import requests
from dotenv import load_dotenv


# Always load .env from the same folder as this Python file
BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"

if not ENV_FILE.exists():
    raise RuntimeError(f".env file not found: {ENV_FILE}")

load_dotenv(dotenv_path=ENV_FILE, override=True)

API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise RuntimeError(
        f"OPENROUTER_API_KEY is missing from {ENV_FILE}"
    )

MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "openai/gpt-4o-mini"
)

API_URL = "https://openrouter.ai/api/v1/chat/completions"


SYSTEM_PROMPT = """
You are EconBizGPT, an AI assistant specialized only in:

- Economics
- Accounting
- Bookkeeping
- Financial statements
- Corporate finance
- Business strategy
- Entrepreneurship
- Marketing and sales
- Operations and supply chains
- Management
- Business mathematics
- Financial analysis
- Inflation, interest rates, GDP, employment, and trade

Use semantic understanding to decide whether a question belongs to these areas.
Do not use keyword matching, regular expressions, or manually coded topic rules.

If the user's question is unrelated to economics, accounting, finance, or business,
politely refuse and say that you only answer questions in those fields.

For accounting questions:
- Explain debit and credit logic.
- Show journal entries when useful.
- Distinguish assets, liabilities, equity, income, and expenses.

For economics and business questions:
- Explain assumptions.
- Use formulas and examples where helpful.
- Distinguish facts, estimates, and opinions.
- Do not invent statistics, tax rules, laws, company information, or financial figures.

For tax, legal, investment, audit, or regulatory topics:
- Give general educational information only.
- Recommend checking with a qualified professional.
- Do not guarantee investment returns.

If current information is required, explain that current data should be verified
using a reliable and up-to-date source.

Answer clearly and concisely using headings, examples, equations, or tables when useful.
"""


def ask_openrouter(messages):
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 1200
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-OpenRouter-Title": "EconBizGPT"
    }

    response = requests.post(
        API_URL,
        headers=headers,
        json=payload,
        timeout=90
    )

    if response.status_code != 200:
        try:
            error_details = response.json()
        except ValueError:
            error_details = response.text

        raise RuntimeError(
            f"OpenRouter returned HTTP {response.status_code}: "
            f"{error_details}"
        )

    result = response.json()

    try:
        return result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(f"Unexpected API response: {result}")


def main():
    conversation = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    print("Susie is ready.")
    print("Ask economics, accounting, finance, or business questions.")
    print("Type exit or quit to stop.\n")

    while True:
        user_message = input("You: ").strip()

        if user_message.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        if not user_message:
            continue

        conversation.append(
            {
                "role": "user",
                "content": user_message
            }
        )

        try:
            answer = ask_openrouter(conversation)

            conversation.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

            print(f"\nSusie: {answer}\n")

        except Exception as error:
            conversation.pop()
            print(f"\nError: {error}\n")


if __name__ == "__main__":
    main()