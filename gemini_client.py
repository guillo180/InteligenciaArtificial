"""System Instruction y función reutilizable para consultar Gemini."""
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()


# ============================================================
# CONFIGURACIÓN
# ============================================================

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)

MODEL = "gemini-3.6-flash"


SYSTEM_INSTRUCTION = (
    "Eres un instructor de programación para principiantes. "
    "Respondes siempre en español. "
    "Utilizas explicaciones sencillas y claras. "
    "No utilizas jerga técnica sin explicarla. "
    "No inventas funciones ni información. "
    "Respondes como máximo en 3 frases."
)


# ============================================================
# CONTAR TOKENS
# ============================================================

def print_budget(contents: list[dict]) -> None:

    tokens = client.models.count_tokens(
        model=MODEL,
        contents=contents,
    )

    print(
        f"Historial: {tokens.total_tokens} tokens"
    )


# ============================================================
# CONSULTAR GEMINI
# ============================================================

def ask(prompt: str) -> tuple[str, str]:

    contents = [
        {
            "role": "user",
            "parts": [
                {
                    "text": prompt
                }
            ],
        }
    ]

    print_budget(contents)

    response = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            max_output_tokens=1000,
        ),
    )

    finish_reason = str(
        response.candidates[0].finish_reason
    )

    if "MAX_TOKENS" in finish_reason:
        print(
            "[warning] La respuesta fue truncada "
            "por max_output_tokens."
        )

    return response.text, finish_reason


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main() -> None:

    text, finish_reason = ask(
        "¿Qué opinas de var en JavaScript?"
    )

    print("\n========== RESPUESTA ==========")
    print(text)

    print("\n========== FINALIZACIÓN ==========")
    print(f"finish: {finish_reason}")


def main() -> None:

    r1_text, _ = ask(
        "Hola, me llamo Valeria."
    )

    print("BOT:", r1_text)

    r2_text, _ = ask(
        "¿Cómo me llamo?"
    )

    print("BOT:", r2_text)


if __name__ == "__main__":
    main()