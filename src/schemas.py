"""
Application data models
"""

from typing import Dict, List, Optional, Any, Union, Literal
from pydantic import BaseModel, Field


class ImageUrl(BaseModel):
    """Image URL model"""
    url: str
    detail: Optional[str] = "auto"


class ContentPart(BaseModel):
    """Content part model for OpenAI's new content format"""
    type: str
    text: Optional[str] = None
    image_url: Optional[ImageUrl] = None


class ToolFunction(BaseModel):
    """Tool function definition"""
    name: str
    description: Optional[str] = None
    parameters: Dict[str, Any]


class Tool(BaseModel):
    """Tool definition"""
    type: Literal["function"]
    function: ToolFunction


class ToolChoice(BaseModel):
    """Tool choice definition"""
    type: Literal["function"]
    function: Dict[str, str]


class Message(BaseModel):
    """Chat message model"""
    role: str
    content: Optional[Union[str, List[ContentPart]]] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    
    class Config:
        extra = "allow"


class OpenAIRequest(BaseModel):
    """OpenAI-compatible request model"""
    model: str
    messages: List[Message]
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, ToolChoice]] = None
    
    class Config:
        extra = "allow"


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

