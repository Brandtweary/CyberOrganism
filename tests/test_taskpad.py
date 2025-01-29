import os
import tempfile
from datetime import datetime, timedelta
import pytest
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models import Task, Tag
from src.store import TaskStore

@pytest.fixture
def temp_store():
    """Create a temporary TaskStore that uses a temporary file."""
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
        store = TaskStore(filepath=tmp.name)
        yield store
        # Cleanup
        os.unlink(tmp.name)

def test_basic_task_operations(temp_store):
    """Test basic task creation and completion."""
    # Create a simple task
    task = Task(title="Test task")
    temp_store.add_task(task)
    
    # Verify task was added
    tasks = temp_store.get_all_tasks()
    assert len(tasks) == 1
    assert tasks[0].title == "Test task"
    assert not tasks[0].completed
    
    # Complete the task
    temp_store.complete_task(task.id)
    
    # Verify task was completed
    tasks = temp_store.get_all_tasks(include_completed=True)  # Include completed tasks
    assert len(tasks) == 1
    assert tasks[0].completed
    assert tasks[0].completed_at is not None

def test_recurring_task(temp_store):
    """Test that recurring tasks are properly handled."""
    # Create a recurring task (weekly)
    task = Task(title="Weekly task")
    recurring_tag = Tag(name="recurring", value="7")
    task.add_tag(recurring_tag)
    temp_store.add_task(task)
    
    # Verify task was added with recurring days
    tasks = temp_store.get_all_tasks()
    assert len(tasks) == 1
    assert tasks[0].recurring_days == 7
    
    # Complete the task
    original_task = tasks[0]
    temp_store.complete_task(original_task.id)
    
    # Verify:
    # 1. Original task is completed
    # 2. New task was created
    # 3. New task is scheduled for 7 days after completion
    all_tasks = temp_store.get_all_tasks(include_completed=True)  # Include completed tasks
    assert len(all_tasks) == 2  # Original + new recurring task
    
    # Find completed and pending tasks
    completed_task = next(t for t in all_tasks if t.completed)
    pending_task = next(t for t in all_tasks if not t.completed)
    
    # Check completed task
    assert completed_task.id == original_task.id
    assert completed_task.completed
    assert completed_task.completed_at is not None
    
    # Check new recurring task
    assert pending_task.id != original_task.id
    assert not pending_task.completed
    assert pending_task.title == "Weekly task"
    assert pending_task.recurring_days == 7
    # Verify scheduled time is 7 days after completion
    expected_time = completed_task.completed_at + timedelta(days=7)
    assert pending_task.created_at == expected_time

def test_recurring_task_with_metadata(temp_store):
    """Test that recurring tasks preserve metadata and tags."""
    # Create a recurring task with metadata and additional tags
    task = Task(
        title="Complex recurring task",
        description="Test description",
        metadata={"priority": "high"}
    )
    task.add_tag(Tag(name="recurring", value="7"))
    task.add_tag(Tag(name="project", value="test"))
    temp_store.add_task(task)
    
    # Complete the task
    original_task = next(iter(temp_store.get_all_tasks()))
    temp_store.complete_task(original_task.id)
    
    # Get the new recurring task
    all_tasks = temp_store.get_all_tasks(include_completed=True)  # Include completed tasks
    pending_task = next(t for t in all_tasks if not t.completed)
    
    # Verify all metadata and tags were preserved
    assert pending_task.description == "Test description"
    assert pending_task.metadata == {"priority": "high"}
    assert len(pending_task.tags) == 2
    assert any(t.name == "recurring" and t.value == "7" for t in pending_task.tags)
    assert any(t.name == "project" and t.value == "test" for t in pending_task.tags)
