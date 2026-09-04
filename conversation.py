"""In-memory conversation history — the model 'remembers' because we resend it."""


import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types, errors


# ============================================================
# CONFIGURACIÓN
# ============================================================

load_dotenv()

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)

MODEL = "gemini-3.6-flash"

SYSTEM_INSTRUCTION = (
    "Eres un asistente breve. "
    "Respondes en español."
)


# ============================================================
# HISTORIAL
# ============================================================

history: list[dict] = []


# ============================================================
# VENTANA DESLIZANTE
# ============================================================

MAX_TURNS = 10
# Mantiene los últimos 10 intercambios:
# 10 mensajes del usuario + 10 respuestas del modelo = 20 entradas.


def trim_history() -> None:
    """
    Mantiene solamente los últimos MAX_TURNS intercambios.
    """

    max_entries = MAX_TURNS * 2

    if len(history) > max_entries:
        del history[:-max_entries]


# ============================================================
# ENVÍO DE MENSAJES
# ============================================================

def send(message: str, _retries: int = 0) -> str:
    """
    Envía un mensaje a Gemini utilizando el historial.

    429:
        Reintenta hasta 3 veces utilizando backoff exponencial.

    5xx:
        Reintenta hasta 3 veces porque son errores del servidor.

    Otros ClientError:
        No se reintentan.
    """

    # Primero recortamos el historial si es necesario.
    trim_history()

    # Agregamos el mensaje del usuario.
    history.append({
        "role": "user",
        "parts": [
            {
                "text": message
            }
        ]
    })

    print(f"\n[INFO] Mensajes enviados a Gemini: {len(history)}")

    for i, item in enumerate(history, start=1):
        print(f"[INFO] {i}. {item['role']}")

    try:

        response = client.models.generate_content(
            model=MODEL,
            contents=history,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                max_output_tokens=1000,
            ),
        )

    # ========================================================
    # ERROR 429 - RESOURCE EXHAUSTED
    # ========================================================

    except errors.ClientError as exc:

        if exc.code == 429 and _retries < 3:

            wait = 2 ** _retries

            print(
                f"[429] Límite alcanzado. "
                f"Reintentando en {wait}s..."
            )

            time.sleep(wait)

            # Quitamos el mensaje que acabamos de agregar
            # para evitar duplicarlo en el siguiente intento.
            history.pop()

            return send(
                message,
                _retries=_retries + 1
            )

        # Si no se puede reintentar, quitamos el mensaje
        # que quedó pendiente en el historial.
        history.pop()

        return (
            f"Error del cliente ({exc.code}): "
            f"{exc.message}. No se reintenta."
        )

    # ========================================================
    # ERROR 5xx - SERVER ERROR
    # ========================================================

    except errors.ServerError as exc:

        if _retries < 3:

            wait = 2 ** _retries

            print(
                f"[{exc.code}] Error temporal del servidor. "
                f"Reintentando en {wait}s..."
            )

            time.sleep(wait)

            # Quitamos el mensaje antes de volver a intentar.
            history.pop()

            return send(
                message,
                _retries=_retries + 1
            )

        # Después de 3 intentos fallidos,
        # quitamos el mensaje pendiente.
        history.pop()

        return (
            f"El servicio no respondió tras varios intentos "
            f"({exc.code})."
        )

    # ========================================================
    # RESPUESTA CORRECTA
    # ========================================================

    finish_reason = str(
        response.candidates[0].finish_reason
    )

    if "MAX_TOKENS" in finish_reason:

        print(
            "[warning] Respuesta truncada por "
            "max_output_tokens."
        )

    # Guardamos la respuesta del modelo.
    history.append({
        "role": "model",
        "parts": [
            {
                "text": response.text
            }
        ]
    })

    return response.text


# ============================================================
# DEMO: OLVIDO POR VENTANA DESLIZANTE
# ============================================================

def demo_forgetting() -> None:
    """
    Demo independiente:
    una ventana pequeña termina olvidando el principio
    de la conversación.

    No forma parte de la ejecución principal.
    """

    global history, MAX_TURNS

    history = []

    original_max = MAX_TURNS

    # Ventana pequeña a propósito.
    MAX_TURNS = 3

    print("\n========================================")
    print(" DEMO: VENTANA DESLIZANTE")
    print(" MAX_TURNS = 3")
    print("========================================")

    print(
        send(
            "Mi mascota se llama Rocko."
        )
    )

    for i in range(1, 7):

        print(
            send(
                f"Pregunta de relleno número {i}."
            )
        )

    print(
        send(
            "¿Cómo se llama mi mascota?"
        )
    )

    # Restauramos la configuración original.
    MAX_TURNS = original_max
    history = []


# ============================================================
# DEMO: PROVOCAR RATE LIMIT
# ============================================================

def trigger_rate_limit() -> None:
    """
    Envía varias solicitudes consecutivas para demostrar
    el manejo del límite de solicitudes.

    Esta función es solamente una demostración del curso.
    """

    global history

    history = []

    print("\n========================================")
    print(" DEMO: RATE LIMIT")
    print("========================================")

    for i in range(1, 21):

        print(
            f"\nRequest {i}: "
            f"{send(f'Cuenta hasta {i}.')}"
        )


# ============================================================
# CONVERSACIÓN PRINCIPAL
# ============================================================

def main() -> None:

    print(
        send(
            "Me llamo Alex y mi color favorito es el verde."
        )
    )

    print(
        send(
            "¿Qué framework de Python vimos en la Clase 1?"
        )
    )

    print(
        send(
            "Dame un ejemplo de dato que no cabe en un int."
        )
    )

    print(
        send(
            "¿Qué hace el comando uv init?"
        )
    )

    print(
        send(
            "Explica en una frase qué es un token."
        )
    )

    print(
        send(
            "¿Qué significa que una API sea stateless?"
        )
    )

    print(
        send(
            "¿Para qué sirve un archivo .env?"
        )
    )

    print(
        send(
            "¿Cómo me llamo y cuál es mi color favorito?"
        )
    )


# if __name__ == "__main__":
#     main()


if __name__ == "__main__":
    trigger_rate_limit()
