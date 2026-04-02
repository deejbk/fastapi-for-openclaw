import logging
from fastapi import FastAPI, HTTPException
from models import ChatRequest, ChatResponse, SummariseRequest, SummariseResponse
from llm_client import call_llm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


@app.get("/")
async def root():
    return {"greeting": "Hello, World!", "message": "Welcome to FastAPI!"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a message to the LLM and return the response.
    """
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": request.message},
        ]
        response = await call_llm(messages)
        return ChatResponse(response=response)
    except ValueError as e:
        logger.error("Chat validation error: %s", str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error("Chat runtime error: %s", str(e))
        raise HTTPException(status_code=502, detail="LLM service unavailable")


@app.post("/summarise", response_model=SummariseResponse)
async def summarise(request: SummariseRequest):
    """
    Summarise the provided text using the LLM.
    """
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {
                "role": "user",
                "content": f"Summarise the following text clearly and concisely:\n\n{request.text}",
            },
        ]
        summary = await call_llm(messages)
        return SummariseResponse(summary=summary)
    except ValueError as e:
        logger.error("Summarise validation error: %s", str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error("Summarise runtime error: %s", str(e))
        raise HTTPException(status_code=502, detail="LLM service unavailable")
