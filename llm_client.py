import os
import logging
import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-3.5-turbo")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"


async def call_llm(messages: list) -> str:
    """
    Call OpenRouter API with the given messages and return the model response.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.

    Returns:
        The model's response content as a string.

    Raises:
        ValueError: If API key is not set or response is invalid.
        RuntimeError: If the API request fails.
    """
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                OPENROUTER_BASE_URL,
                headers=headers,
                json=payload,
                timeout=15.0,
            )

            if response.status_code != 200:
                logger.error(
                    "OpenRouter API returned status %d: %s",
                    response.status_code,
                    response.text,
                )
                raise RuntimeError(
                    f"OpenRouter API request failed with status {response.status_code}"
                )

            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content")

            if not content:
                logger.error("Empty response from OpenRouter API: %s", data)
                raise ValueError("Received empty response from LLM")

            return content

        except httpx.TimeoutException:
            logger.error("OpenRouter API request timed out")
            raise RuntimeError("LLM request timed out")
        except httpx.RequestError as e:
            logger.error("OpenRouter API request error: %s", str(e))
            raise RuntimeError(f"LLM request failed: {str(e)}")