from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Set, Dict, Any
from uuid import UUID, uuid4

@dataclass(frozen=True)
class Tag:
    name: str
    value: Optional[Any] = None
    schema: Optional[str] = None

@dataclass
class Task:
    id: UUID = field(default_factory=uuid4)
    title: str = ""
    description: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now())
    tags: Set[Tag] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    completed: bool = field(default=False)
    completed_at: Optional[datetime] = field(default=None)

    def add_tag(self, tag: Tag) -> None:
        self.tags.add(tag)

    def remove_tag(self, tag_name: str) -> None:
        self.tags = {tag for tag in self.tags if tag.name != tag_name}

    def has_tag(self, tag_name: str) -> bool:
        return any(tag.name == tag_name for tag in self.tags)

    def complete(self) -> None:
        self.completed = True
        self.completed_at = datetime.now()

@dataclass
class Note:
    id: UUID = field(default_factory=uuid4)
    content: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now())
    tags: Set[Tag] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_tag(self, tag: Tag) -> None:
        self.tags.add(tag)

    def remove_tag(self, tag_name: str) -> None:
        self.tags = {tag for tag in self.tags if tag.name != tag_name}

    def has_tag(self, tag_name: str) -> bool:
        return any(tag.name == tag_name for tag in self.tags)
