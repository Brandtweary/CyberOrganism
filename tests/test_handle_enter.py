#!/usr/bin/env python3
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Set, Dict, Any
from uuid import UUID, uuid4
from datetime import datetime
from pynput import keyboard
from pynput.keyboard import Key, Listener

# Mock classes to simulate the app environment
class InputMode(Enum):
    TASK = auto()
    NOTE = auto()
    EDIT = auto()
    COMPLETE = auto()
    COMMAND = auto()

@dataclass
class Tag:
    name: str = ""

@dataclass
class Task:
    id: UUID = field(default_factory=uuid4)
    title: str = ""
    completed: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now())
    tags: Set[Tag] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_tag(self, tag: Tag) -> None:
        self.tags.add(tag)

    def complete(self) -> None:
        self.completed = True

@dataclass
class Note:
    id: UUID = field(default_factory=uuid4)
    content: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now())
    tags: Set[Tag] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_tag(self, tag: Tag) -> None:
        self.tags.add(tag)

class MockBuffer:
    def __init__(self):
        self.text = ""
        self.reset_called = False
    
    def reset(self):
        self.reset_called = True
        self.text = ""

class MockApp:
    def __init__(self):
        self.mode = InputMode.TASK
        self.command_mode = False
        self.input_buffer = MockBuffer()
        self.selected_index = 0
        self.seen_commands = False
        self.tasks = []
        self.notes = []
        self.log_messages = []
        self.invalidate_called = False
        self.app = self  # Reference to self for app.app.invalidate() calls
    
    def log_message(self, msg: str, level: str = 'info') -> None:
        self.log_messages.append((msg, level))
    
    def handle_command(self, cmd: str) -> None:
        self.log_messages.append(f"Command handled: {cmd}")

    def invalidate(self) -> None:
        self.invalidate_called = True

    def get_all_tasks(self) -> list[Task]:
        return self.tasks

    def get_all_notes(self) -> list[Note]:
        return self.notes

    def add_task(self, task: Task) -> None:
        self.tasks.append(task)

    def add_note(self, note: Note) -> None:
        self.notes.append(note)

    def _save(self) -> None:
        pass

def process_hashtags(text):
    """Mock hashtag processing."""
    tags = []
    cleaned_text = text
    return cleaned_text, tags

class MockKeyboardState:
    """Mock keyboard state tracker."""
    def __init__(self):
        self.ctrl_pressed = False
        
        def on_press(key):
            if key == Key.ctrl:
                self.ctrl_pressed = True
        
        def on_release(key):
            if key == Key.ctrl:
                self.ctrl_pressed = False
        
        self.listener = Listener(
            on_press=on_press,
            on_release=on_release)
        self.listener.start()
    
    def stop(self):
        """Stop the keyboard listener."""
        self.listener.stop()

def handle_enter(event, app, kb_state):
    """Isolated handle_enter function for testing."""
    text = app.input_buffer.text

    # Mark help as seen since user is entering a command
    app.seen_commands = True

    # Check for slash command in any mode
    if text.startswith('/'):
        app.handle_command(text[1:])  # Strip the slash
    elif app.command_mode:
        app.handle_command(text)
    else:
        if app.mode == InputMode.TASK:
            cleaned_text, tags = process_hashtags(text)
            task = Task(title=cleaned_text)
            for tag in tags:
                task.add_tag(Tag(name=tag))
            app.add_task(task)
            app.log_message(f"Added new task: {cleaned_text}")
        elif app.mode == InputMode.NOTE:
            cleaned_text, tags = process_hashtags(text)
            note = Note(content=cleaned_text)
            for tag in tags:
                note.add_tag(Tag(name=tag))
            app.add_note(note)
            app.log_message(f"Added new note: {cleaned_text}")
        elif app.mode == InputMode.COMPLETE:
            try:
                task_num = int(text)
                tasks = app.get_all_tasks()
                
                if task_num >= len(tasks) or task_num < 0:
                    app.log_message("Invalid task number", 'warning')
                else:
                    task = tasks[task_num]
                    task.complete()
                    app._save()
                    app.log_message(f"Marked task {task_num} as completed")
            except ValueError:
                app.log_message("Please enter a valid task number", 'warning')
        elif app.mode == InputMode.EDIT:
            # Get all entries in display order
            entries = []
            entries.append(None)  # Placeholder for new task slot
            entries.extend(app.get_all_tasks())
            entries.extend(app.get_all_notes())

            # Get the selected entry
            if app.selected_index >= len(entries):
                return

            entry = entries[app.selected_index]

            # Check for ctrl+enter first
            if kb_state.ctrl_pressed:
                if isinstance(entry, Task):
                    entry.complete()
                    app._save()
                    app.log_message(f"Marked task {app.selected_index} as completed")
                    app.input_buffer.reset()
                return

            # Handle normal edit mode
            cleaned_text, tags = process_hashtags(text)
            if entry is None:
                task = Task(title=cleaned_text)
                for tag in tags:
                    task.add_tag(Tag(name=tag))
                app.add_task(task)
                app.log_message(f"Added new task: {cleaned_text}")
                app.input_buffer.reset()
            else:
                if isinstance(entry, Task):
                    entry.title = cleaned_text
                    entry.tags.clear()
                    for tag in tags:
                        entry.add_tag(Tag(name=tag))
                    app.log_message(f"Updated task {app.selected_index}")
                else:  # Note
                    entry.content = cleaned_text
                    entry.tags.clear()
                    for tag in tags:
                        entry.add_tag(Tag(name=tag))
                    app.log_message(f"Updated note {app.selected_index}")

    # Only clear buffer if we're not in edit mode or if we just created a new task
    if app.mode == InputMode.TASK or app.mode == InputMode.NOTE or app.mode == InputMode.COMPLETE or (app.mode == InputMode.EDIT and app.selected_index == 0):
        app.input_buffer.reset()
    app.invalidate()

def test_handle_enter():
    """Test various scenarios for handle_enter."""
    # Create mock app and keyboard state
    app = MockApp()
    kb_state = MockKeyboardState()
    
    try:
        # Test 1: Basic task creation
        print("\nTest 1: Basic task creation")
        app.mode = InputMode.TASK
        app.input_buffer.text = "Test task"
        handle_enter(None, app, kb_state)
        assert len(app.tasks) == 1, "Task should be created"
        assert app.tasks[0].title == "Test task", "Task title should match"
        assert app.input_buffer.reset_called, "Buffer should be reset"
        
        # Test 2: Edit mode - new task
        print("\nTest 2: Edit mode - new task")
        app.mode = InputMode.EDIT
        app.selected_index = 0
        app.input_buffer.text = "New task in edit"
        app.input_buffer.reset_called = False
        handle_enter(None, app, kb_state)
        assert len(app.tasks) == 2, "New task should be created in edit mode"
        assert app.input_buffer.reset_called, "Buffer should be reset"
        
        # Test 3: Edit mode - ctrl+enter on task
        print("\nTest 3: Edit mode - ctrl+enter on task")
        app.mode = InputMode.EDIT
        app.selected_index = 1  # Select the first actual task
        app.input_buffer.text = "Some text"
        app.input_buffer.reset_called = False
        
        # Simulate ctrl press
        kb_state.ctrl_pressed = True
        handle_enter(None, app, kb_state)
        assert app.tasks[0].completed, "Task should be completed"
        assert app.input_buffer.reset_called, "Buffer should be reset"
        kb_state.ctrl_pressed = False
    
    finally:
        # Clean up
        kb_state.stop()

if __name__ == '__main__':
    test_handle_enter()
