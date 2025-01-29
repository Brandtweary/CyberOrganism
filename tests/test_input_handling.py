import os
import tempfile
from datetime import datetime
import pytest
import sys
import threading
import time
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from src.app import CyberOrganism, InputMode
from src.models import Task, Note, Tag
from src.store import TaskStore

@pytest.fixture(autouse=True)
def clean_store():
    """Create a fresh store file for each test."""
    store_file = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
    store_file.close()
    old_store = os.environ.get('CYBERORGANISM_STORE')
    os.environ['CYBERORGANISM_STORE'] = store_file.name
    yield store_file.name  # Return the store file path
    if old_store:
        os.environ['CYBERORGANISM_STORE'] = old_store
    else:
        del os.environ['CYBERORGANISM_STORE']
    os.unlink(store_file.name)

@pytest.fixture
def mock_input():
    """Create a pipe input for testing."""
    with create_pipe_input() as pipe_input:
        with create_app_session(input=pipe_input, output=DummyOutput()):
            yield pipe_input

@pytest.fixture
def app(clean_store):
    """Create a CyberOrganism instance."""
    # Create app instance with the store file path
    app = CyberOrganism(store=TaskStore(clean_store))
    
    # Start app in a separate thread
    thread = threading.Thread(target=app.run)
    thread.daemon = True  # Thread will be killed when test exits
    thread.start()
    
    # Wait a bit for app to start
    time.sleep(0.1)
    
    yield app
    
    # Stop the app
    app.running = False
    thread.join(timeout=1)

def send_text(mock_input, app, text):
    """Helper to send text to the app."""
    app.input_buffer.text = text
    mock_input.send_text('\n')  # Simulate Enter key
    time.sleep(0.1)  # Give app time to process input

def test_basic_task_input(mock_input, app):
    """Test basic task input handling."""
    # Start in task mode
    app.mode = InputMode.TASK
    app.command_mode = False
    
    # Send input
    send_text(mock_input, app, "Test task #important")
    
    # Verify task was created
    tasks = app.store.get_all_tasks()
    assert len(tasks) == 1
    task = tasks[0]
    assert task.title == "Test task"
    assert any(tag.name == "important" for tag in task.tags)

def test_note_input(mock_input, app):
    """Test note input handling."""
    # Start in note mode
    app.mode = InputMode.NOTE
    app.command_mode = False
    
    # Send input
    send_text(mock_input, app, "Test note #[[category: test]]")
    
    # Verify note was created
    notes = app.store.get_all_notes()
    assert len(notes) == 1
    note = notes[0]
    assert note.content == "Test note"
    assert any(tag.name == "category" and tag.value == "test" for tag in note.tags)

def test_command_mode(mock_input, app):
    """Test command mode handling."""
    # Start in command mode
    app.mode = InputMode.TASK
    app.command_mode = True
    
    # Send command
    send_text(mock_input, app, "note")
    
    # Verify mode was switched
    assert app.mode == InputMode.NOTE
    assert not app.command_mode

def test_slash_command(mock_input, app):
    """Test slash command handling."""
    # Start in task mode
    app.mode = InputMode.TASK
    app.command_mode = False
    
    # Send slash command
    send_text(mock_input, app, "/note")
    
    # Verify mode was switched
    assert app.mode == InputMode.NOTE
    assert not app.command_mode

def test_edit_mode_new_task(mock_input, app):
    """Test creating a new task in edit mode."""
    # Start in edit mode
    app.mode = InputMode.EDIT
    app.command_mode = False
    app.selected_index = 0  # New task slot
    
    # Send input
    send_text(mock_input, app, "New task in edit mode #priority")
    
    # Verify task was created
    tasks = app.store.get_all_tasks()
    assert len(tasks) == 1
    task = tasks[0]
    assert task.title == "New task in edit mode"
    assert any(tag.name == "priority" for tag in task.tags)

def test_edit_mode_update_task(mock_input, app):
    """Test updating an existing task in edit mode."""
    # Create a task first
    app.mode = InputMode.TASK
    send_text(mock_input, app, "Original title")
    
    # Switch to edit mode and select the task
    app.mode = InputMode.EDIT
    app.command_mode = False
    app.selected_index = 1  # Skip new task slot
    
    # Send update
    send_text(mock_input, app, "Updated title #updated")
    
    # Verify task was updated
    tasks = app.store.get_all_tasks()
    assert len(tasks) == 1
    updated_task = tasks[0]
    assert updated_task.title == "Updated title"
    assert any(tag.name == "updated" for tag in updated_task.tags)

def test_complete_task_command(mock_input, app):
    """Test completing a task via command."""
    # Create a task first
    app.mode = InputMode.TASK
    send_text(mock_input, app, "Task to complete")
    
    # Switch to command mode
    app.mode = InputMode.TASK
    app.command_mode = True
    
    # Send complete command
    send_text(mock_input, app, "complete 0")  # Complete first task
    
    # Verify task was completed
    tasks = app.store.get_all_tasks(include_completed=True)
    assert len(tasks) == 1
    completed_task = tasks[0]
    assert completed_task.completed
    assert completed_task.completed_at is not None
