"""
Application data models
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel


class ContentPart(BaseModel):
    """Content part model for OpenAI's new content format"""
    type: str
    text: Optional[str] = None


class Message(BaseModel):
    """Chat message model"""
    role: str
    content: Optional[Union[str, List[ContentPart]]] = None
    reasoning_content: Optional[str] = None


class OpenAIRequest(BaseModel):
    """OpenAI-compatible request model"""
    model: str
    messages: List[Message]
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class Model(BaseModel):
    """Model information for listing"""
    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelsResponse(BaseModel):
    """Models list response model"""
    object: str = "list"
    data: List[Model]

