from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Literal, Set
from cymbiont_logger.process_log import ProcessLog
from datetime import datetime
import asyncio

MessageRole = Literal["user", "assistant", "system"]

@dataclass
class ChatMessage:
    role: MessageRole
    content: str
    name: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

class LLM(Enum):
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    O1_PREVIEW = "o1-preview"
    SONNET_3_5 = "claude-3-5-sonnet-latest"
    HAIKU_3_5 = "claude-3-5-haiku-latest"

@dataclass
class TokenUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int
    timestamp: float

@dataclass
class SystemMessagePart:
    header: str
    content: str
    required_params: List[str]

@dataclass
class SystemPromptPartInfo:
    """Info for a single system prompt part"""
    toggled: bool
    index: int

@dataclass
class SystemPromptPartsData:
    """Data structure for system prompt parts configuration.
    Each part has a toggle state and an index for ordering."""
    parts: Dict[str, SystemPromptPartInfo]
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Validate that all values are SystemPromptPartInfo
        for part_name, part_info in self.parts.items():
            if not isinstance(part_info, SystemPromptPartInfo):
                if isinstance(part_info, dict):
                    self.parts[part_name] = SystemPromptPartInfo(**part_info)
                else:
                    raise ValueError(f"Invalid part info for {part_name}: {part_info}")

    def add_part(self, name: str, info: SystemPromptPartInfo, **kwargs):
        """Add a new part with its kwargs"""
        self.parts[name] = info
        self.kwargs.update(kwargs)

class ToolName(Enum):
    CONTEMPLATE_LOOP = "contemplate_loop"
    EXIT_LOOP = "exit_loop"
    MESSAGE_SELF = "message_self"
    EXECUTE_SHELL_COMMAND = "execute_shell_command"
    TOGGLE_PROMPT_PART = "toggle_prompt_part"
    INTRODUCE_SELF = "introduce_self"
    SHELL_LOOP = "shell_loop"
    MEDITATE = "meditate"

class ToolChoice(Enum):
    """Tool choice options for API calls."""
    AUTO = "auto"
    REQUIRED = "required"
    NONE = "none"

    def to_literal(self) -> Literal["auto", "required", "none"]:
        """Convert enum value to literal type expected by API."""
        return self.value  # type: ignore

@dataclass
class ToolLoopData:
    """Data for managing tool loops."""
    loop_type: str
    loop_message: str
    active: bool = True
    available_tools: Set[ToolName] = field(default_factory=set)
    loop_tokens: int = 0
    system_prompt_parts: Optional[SystemPromptPartsData] = None

@dataclass
class APICall:
    model: str
    messages: List[ChatMessage]
    system_message: str
    timestamp: float
    mock: bool
    mock_tokens: Optional[int]
    expiration_counter: int
    future: asyncio.Future[Dict[str, Any]]
    provider: str
    max_completion_tokens: int
    temperature: float = 0.7
    process_log: Optional[ProcessLog] = None
    tools: Optional[Set[ToolName]] = None
    system_prompt_parts: Optional[SystemPromptPartsData] = None
    tool_choice: Literal["auto", "required", "none"] = "auto"