"""
Keyboard bindings for the CyberOrganism application.
"""

from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.filters import Condition
from pynput import keyboard

from .models import Task, Note, Tag
from .commands import process_hashtags
from .logger import setup_logger

logger = setup_logger()

def create_keybindings(app) -> KeyBindings:
    """Create and return the keybindings for the application."""
    kb = KeyBindings()
    
    # Track ctrl state
    ctrl_pressed = False
    
    def on_press(key):
        nonlocal ctrl_pressed
        if key == keyboard.Key.ctrl:
            ctrl_pressed = True
            logger.debug("Ctrl pressed")
    
    def on_release(key):
        nonlocal ctrl_pressed
        if key == keyboard.Key.ctrl:
            ctrl_pressed = False
            logger.debug("Ctrl released")
    
    # Start keyboard listener
    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release)
    listener.start()
    app.kb_listener = listener  # Store reference for cleanup

    @kb.add('escape', eager=True)
    def handle_escape(event):
        """Toggle command mode."""
        app.command_mode = not app.command_mode
        if app.command_mode:
            app.seen_commands = True
            if app.mode == app.InputMode.EDIT:
                app.mode = app.InputMode.COMMAND  # Properly exit edit mode
            app.input_buffer.reset()
        app.app.invalidate()

    @kb.add('c-c')
    def handle_exit(event):
        """Exit the application."""
        app.running = False
        event.app.exit()

    @kb.add('c-u')
    def handle_focus_prev(event):
        """Focus previous item."""
        event.app.layout.focus_previous()

    @kb.add('c-d')
    def handle_focus_next(event):
        """Focus next item."""
        event.app.layout.focus_next()

    @kb.add('up', filter=Condition(lambda: app.mode == app.InputMode.EDIT and not app.command_mode))
    def handle_up(event):
        """Move selection up in edit mode."""
        total_items = len(app.store.get_all_tasks()) + len(app.store.get_all_notes()) + 1
        if total_items > 0:
            app.selected_index = (app.selected_index - 1) % total_items
            app._prefill_selected_content()
            app.app.invalidate()

    @kb.add('down', filter=Condition(lambda: app.mode == app.InputMode.EDIT and not app.command_mode))
    def handle_down(event):
        """Move selection down in edit mode."""
        total_items = len(app.store.get_all_tasks()) + len(app.store.get_all_notes()) + 1
        if total_items > 0:
            app.selected_index = (app.selected_index + 1) % total_items
            app._prefill_selected_content()
            app.app.invalidate()

    def complete_selected_task():
        """Complete the currently selected task in edit mode."""
        # Get all entries in display order
        entries = []
        if app.mode == app.InputMode.EDIT:
            entries.append(None)  # Placeholder for new task slot
        entries.extend(app.store.get_all_tasks())
        entries.extend(app.store.get_all_notes())

        # Get the selected entry
        if app.selected_index >= len(entries):
            return

        entry = entries[app.selected_index]
        if entry is None:
            app.log_message("Cannot complete new task slot", 'warning')
        elif not isinstance(entry, Task):
            app.log_message("Cannot complete a note", 'warning')
        else:
            entry.complete()
            app.store._save()
            app.log_message(f"Marked task {app.selected_index} as completed")
            app.input_buffer.reset()
            app.app.invalidate()

    @kb.add('enter', eager=True)
    def handle_enter(event):
        """Process input when enter is pressed."""
        logger.debug("Enter pressed")
        text = app.input_buffer.text
        logger.debug(f"Buffer text: {text!r}")
        logger.debug(f"Current mode: {app.mode}")
        logger.debug(f"Command mode: {app.command_mode}")
        logger.debug(f"Selected index: {app.selected_index}")
        logger.debug(f"Ctrl state: {ctrl_pressed}")

        # Mark help as seen since user is entering a command
        app.seen_commands = True

        # Check for slash command in any mode
        if text.startswith('/'):
            logger.debug("Handling as slash command")
            app.handle_command(text[1:])  # Strip the slash
        elif app.command_mode:
            logger.debug("Handling as command")
            app.handle_command(text)
        else:
            if app.mode == app.InputMode.TASK:
                logger.debug("Processing task creation")
                cleaned_text, tags = process_hashtags(text)
                logger.debug(f"Cleaned text: {cleaned_text!r}, tags: {tags}")
                task = Task(title=cleaned_text)
                for tag in tags:
                    task.add_tag(Tag(name=tag))
                logger.debug("Adding task to store")
                app.store.add_task(task)
                app.log_message(f"Added new task: {cleaned_text}")
            elif app.mode == app.InputMode.NOTE:
                logger.debug("Processing note creation")
                cleaned_text, tags = process_hashtags(text)
                note = Note(content=cleaned_text)
                for tag in tags:
                    note.add_tag(Tag(name=tag))
                app.store.add_note(note)
                app.log_message(f"Added new note: {cleaned_text}")
            elif app.mode == app.InputMode.COMPLETE:
                logger.debug("Processing task completion")
                try:
                    task_num = int(text)
                    tasks = app.store.get_all_tasks()
                    
                    if task_num >= len(tasks) or task_num < 0:
                        app.log_message("Invalid task number", 'warning')
                    else:
                        task = tasks[task_num]
                        task.complete()
                        app.store._save()
                        app.log_message(f"Marked task {task_num} as completed")
                except ValueError:
                    app.log_message("Please enter a valid task number", 'warning')
            elif app.mode == app.InputMode.EDIT:
                logger.debug("Processing edit mode")
                # Get all entries in display order
                entries = []
                entries.append(None)  # Placeholder for new task slot
                entries.extend(app.store.get_all_tasks())
                entries.extend(app.store.get_all_notes())

                # Get the selected entry
                if app.selected_index >= len(entries):
                    logger.debug(f"Invalid selected_index {app.selected_index} >= {len(entries)}")
                    return

                entry = entries[app.selected_index]
                logger.debug(f"Selected entry: {entry}")

                # Check for ctrl+enter first
                if ctrl_pressed:
                    logger.debug("Ctrl+Enter detected")
                    if isinstance(entry, Task):
                        logger.debug("Completing task")
                        entry.complete()
                        app.store._save()
                        app.log_message(f"Marked task {app.selected_index} as completed")
                        app.input_buffer.reset()
                    return

                # Handle normal edit mode
                cleaned_text, tags = process_hashtags(text)
                logger.debug(f"Cleaned text: {cleaned_text!r}, tags: {tags}")
                if entry is None:
                    logger.debug("Creating new task in edit mode")
                    task = Task(title=cleaned_text)
                    for tag in tags:
                        task.add_tag(Tag(name=tag))
                    app.store.add_task(task)
                    app.log_message(f"Added new task: {cleaned_text}")
                    app.input_buffer.reset()
                else:
                    if isinstance(entry, Task):
                        logger.debug("Updating existing task")
                        entry.title = cleaned_text
                        entry.tags.clear()
                        for tag in tags:
                            entry.add_tag(Tag(name=tag))
                        app.log_message(f"Updated task {app.selected_index}")
                    else:  # Note
                        logger.debug("Updating existing note")
                        entry.content = cleaned_text
                        entry.tags.clear()
                        for tag in tags:
                            entry.add_tag(Tag(name=tag))
                        app.log_message(f"Updated note {app.selected_index}")

        # Only clear buffer if we're not in edit mode or if we just created a new task
        logger.debug(f"Checking if buffer should be cleared - mode: {app.mode}, selected_index: {app.selected_index}")
        if app.mode == app.InputMode.TASK or app.mode == app.InputMode.NOTE or app.mode == app.InputMode.COMPLETE or (app.mode == app.InputMode.EDIT and app.selected_index == 0):
            logger.debug("Clearing buffer")
            app.input_buffer.reset()
        app.app.invalidate()
        logger.debug("---End of handle_enter---\n")

    return kb
