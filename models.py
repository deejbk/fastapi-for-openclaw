from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


class SummariseRequest(BaseModel):
    text: str


class SummariseResponse(BaseModel):
    summary: str