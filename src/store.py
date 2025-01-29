from typing import Dict, List, Optional, Union
from uuid import UUID
import json
import os
from datetime import datetime

from .models import Task, Note, Tag

class TaskStore:
    def __init__(self, filepath: str = "cyberorganism.json"):
        self.filepath = filepath
        self.tasks: Dict[UUID, Task] = {}
        self.notes: Dict[UUID, Note] = {}
        self._load()

    def _save(self) -> None:
        data = {
            "tasks": {str(id): self._serialize_item(task) for id, task in self.tasks.items()},
            "notes": {str(id): self._serialize_item(note) for id, note in self.notes.items()}
        }
        with open(self.filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def _load(self) -> None:
        if not os.path.exists(self.filepath):
            return
        
        try:
            with open(self.filepath, 'r') as f:
                data = json.load(f)
                
            for task_id, task_data in data.get("tasks", {}).items():
                self.tasks[UUID(task_id)] = self._deserialize_task(task_data)
                
            for note_id, note_data in data.get("notes", {}).items():
                self.notes[UUID(note_id)] = self._deserialize_note(note_data)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error loading data: {e}")
            # Start fresh if file is corrupted
            self.tasks = {}
            self.notes = {}

    def _serialize_item(self, item: Union[Task, Note]) -> dict:
        tags = [{
            "name": tag.name,
            "value": tag.value,
            "schema": tag.schema
        } for tag in item.tags]
        
        base_data = {
            "id": str(item.id),
            "created_at": item.created_at.isoformat(),
            "tags": tags,
            "metadata": item.metadata,
        }
        
        if isinstance(item, Task):
            base_data.update({
                "title": item.title,
                "description": item.description,
                "completed": item.completed,
            })
            if item.completed_at:
                base_data["completed_at"] = item.completed_at.isoformat()
        else:
            base_data["content"] = item.content
            
        return base_data

    def _deserialize_task(self, data: dict) -> Task:
        tags = {Tag(**tag_data) for tag_data in data.get("tags", [])}
        task = Task(
            id=UUID(data["id"]),
            title=data.get("title", ""),
            description=data.get("description", ""),
            created_at=datetime.fromisoformat(data["created_at"]),
            tags=tags,
            metadata=data.get("metadata", {}),
        )
        # Load completed status
        if data.get("completed", False):
            task.completed = True
            if "completed_at" in data:
                task.completed_at = datetime.fromisoformat(data["completed_at"])
        return task

    def _deserialize_note(self, data: dict) -> Note:
        tags = {Tag(**tag_data) for tag_data in data.get("tags", [])}
        return Note(
            id=UUID(data["id"]),
            content=data.get("content", ""),
            created_at=datetime.fromisoformat(data["created_at"]),
            tags=tags,
            metadata=data.get("metadata", {})
        )

    def add_task(self, task: Task) -> None:
        self.tasks[task.id] = task
        self._save()

    def add_note(self, note: Note) -> None:
        self.notes[note.id] = note
        self._save()

    def get_task(self, task_id: UUID) -> Optional[Task]:
        return self.tasks.get(task_id)

    def get_note(self, note_id: UUID) -> Optional[Note]:
        return self.notes.get(note_id)

    def remove_task(self, task_id: UUID) -> None:
        if task_id in self.tasks:
            del self.tasks[task_id]
            self._save()

    def remove_note(self, note_id: UUID) -> None:
        if note_id in self.notes:
            del self.notes[note_id]
            self._save()

    def get_tasks_by_tag(self, tag_name: str, include_completed: bool = False) -> List[Task]:
        tasks = [task for task in self.tasks.values() if task.has_tag(tag_name)]
        if not include_completed:
            tasks = [task for task in tasks if not task.completed]
        return tasks

    def get_notes_by_tag(self, tag_name: str) -> List[Note]:
        return [note for note in self.notes.values() if note.has_tag(tag_name)]

    def get_all_tasks(self, include_completed: bool = False) -> List[Task]:
        tasks = list(self.tasks.values())
        if not include_completed:
            tasks = [task for task in tasks if not task.completed]
        return tasks

    def get_all_notes(self) -> List[Note]:
        return list(self.notes.values())

    def get_all_items_by_tag(self, tag_name: str, include_completed: bool = False) -> List[Union[Task, Note]]:
        tasks = self.get_tasks_by_tag(tag_name, include_completed)
        return tasks + self.get_notes_by_tag(tag_name)
