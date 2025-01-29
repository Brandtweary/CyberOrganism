from typing import Dict, List, Tuple
from .models import Task, Note, Tag

COMMANDS = {
    "task": "Enter task mode - new input will be saved as tasks",
    "note": "Enter note mode - new input will be saved as notes",
    "edit": "Enter edit mode - modify existing items or add new tasks",
    "tag": "Add tag to item (format: tag <item_number> <tag1> <tag2> <[[multi-word-tag]]>)",
    "complete": "Mark a task as completed (format: complete <task_number>)",
    "exit": "Exit the program"
}

def process_hashtags(text: str) -> tuple[str, set[Tag]]:
    """Extract hashtags from text and clean up the content.
    Returns (cleaned_text, set_of_tags)
    
    Supports three tag formats:
    1. Simple tags: #tag
    2. Multi-word tags: #[[multi word tag]]
    3. Property tags: #[[tag: value]]
    """
    tags = set()
    content_words = []
    
    # Handle multi-word tags first
    import re
    # Find all instances of #[[...]]
    bracket_pattern = r'#\[\[(.*?)\]\]'
    # Replace multi-word tags with placeholders and collect the tags
    placeholder_counter = 0
    placeholders = {}
    
    def replace_tag(match):
        nonlocal placeholder_counter
        tag_content = match.group(1).strip()
        if tag_content:  # Only process non-empty tags
            # Check if it's a property tag (contains :)
            if ':' in tag_content:
                name, value = tag_content.split(':', 1)
                name = name.strip()
                value = value.strip()
                if name:  # Only add if name is non-empty
                    tags.add(Tag(name=name, value=value))
            else:
                tags.add(Tag(name=tag_content))
            placeholder = f"__TAG_PLACEHOLDER_{placeholder_counter}__"
            placeholders[placeholder] = tag_content
            placeholder_counter += 1
            return placeholder
        return ""
        
    processed_text = re.sub(bracket_pattern, replace_tag, text)
    
    # Now process the remaining text for simple hashtags
    words = processed_text.split()
    if not words:
        return "", tags
        
    # Process all words except the last one
    for word in words[:-1]:
        if word.startswith('#'):
            # Remove the # but keep the word content, and add to tags
            tag = word[1:]
            if tag:  # Only add non-empty tags
                tags.add(Tag(name=tag))
                # For inline tags, keep the tag content
                content_words.append(tag)
        elif word in placeholders:
            # Restore multi-word tag content
            content_words.append(placeholders[word])
        else:
            content_words.append(word)
    
    # Special handling for the last word
    last_word = words[-1] if words else ""
    if last_word.startswith('#'):
        # If it's a hashtag at the end, treat it as a pure tag
        tag = last_word[1:]
        if tag:
            tags.add(Tag(name=tag))
    elif last_word in placeholders:
        # For a multi-word tag at the end, treat it as a pure tag
        # Don't add to content_words
        pass
    elif last_word:
        content_words.append(last_word)
    
    return ' '.join(content_words), tags

def handle_command(app_state, command: str) -> None:
    cmd = command.lower().strip()
    app_state.status_message = None  # Clear previous message
    
    if cmd == "exit":
        app_state.running = False
        app_state.app.exit()
    elif cmd == "task":
        app_state.mode = app_state.InputMode.TASK
        app_state.command_mode = False
        app_state.log_message("Switched to TASK mode")
    elif cmd == "note":
        app_state.mode = app_state.InputMode.NOTE
        app_state.command_mode = False
        app_state.log_message("Switched to NOTE mode")
    elif cmd == "edit":
        app_state.mode = app_state.InputMode.EDIT
        app_state.command_mode = False
        # Always start with the "new task" slot selected
        app_state.selected_index = 0
        app_state._prefill_selected_content()
        app_state.app.invalidate()
        app_state.log_message(f"Switched to EDIT mode")
    elif cmd == "command":  # Hidden command to enter command mode
        app_state.command_mode = True
        app_state.app.invalidate()
    elif command.startswith("complete"):
        # Check if a task number was provided
        parts = command.strip().split()
        if len(parts) == 1:
            # Just "complete" - switch to complete mode
            app_state.mode = app_state.InputMode.COMPLETE
            app_state.command_mode = False
            app_state.log_message("Switched to COMPLETE mode - Enter task number to complete")
        elif len(parts) == 2:
            # "complete <number>" - complete the task directly
            try:
                task_num = int(parts[1])
                tasks = app_state.store.get_all_tasks()
                
                if task_num >= len(tasks) or task_num < 0:
                    app_state.log_message("Invalid task number", 'warning')
                else:
                    task = tasks[task_num]
                    task.complete()
                    app_state.store._save()
                    app_state.log_message(f"Marked task {task_num} as completed")
                    app_state.app.invalidate()
            except ValueError:
                app_state.log_message("Invalid task number", 'warning')
        else:
            app_state.log_message("Use: complete [task_number]", 'warning')
    elif command.startswith("tag"):
        try:
            # Split only on the first space to get the command and the rest
            cmd_parts = command.split(" ", 1)
            if len(cmd_parts) < 2:
                raise ValueError("Missing arguments")
            
            # Split the rest on the first space to get item_num and tags
            rest = cmd_parts[1].strip()
            try:
                item_num_str, tags_str = rest.split(" ", 1)
                item_num = int(item_num_str)
            except ValueError:
                raise ValueError("Invalid item number")
            
            entries = []
            if app_state.mode == app_state.InputMode.EDIT:
                entries.append(None)  # Placeholder for new task slot
            entries.extend(app_state.store.get_all_tasks())
            entries.extend(app_state.store.get_all_notes())
            
            if item_num >= len(entries) or item_num < 0:
                raise ValueError("Invalid item number")
            
            entry = entries[item_num]
            if entry is None:
                raise ValueError("Cannot add tags to new task slot")
            
            # Process tags - handle both multi-word [[tags]] and single word tags
            import re
            tags = []
            
            # First extract any [[multi word tags]]
            bracket_pattern = r'\[\[(.*?)\]\]'  # Updated pattern
            # Replace multi-word tags with placeholders and collect them
            placeholder_counter = 0
            placeholders = {}
            
            def replace_tag(match):
                nonlocal placeholder_counter
                tag = match.group(1).strip()
                if tag:
                    placeholder = f"__TAG_PLACEHOLDER_{placeholder_counter}__"
                    placeholders[placeholder] = tag
                    placeholder_counter += 1
                    return placeholder
                return ""
                
            processed_str = re.sub(bracket_pattern, replace_tag, tags_str)
            
            # Now split remaining text on spaces and process each tag
            for tag in processed_str.split():
                if tag in placeholders:
                    # Restore multi-word tag
                    tag = placeholders[tag]
                tags.append(tag.strip())
            
            if not tags:
                raise ValueError("No valid tags provided")
            
            # Add tags to the selected entry
            for tag in tags:
                entry.add_tag(Tag(name=tag))
            
            tag_list = ", ".join(f"'{t}'" for t in tags)
            entry_type = "task" if isinstance(entry, Task) else "note"
            app_state.log_message(f"Added tags {tag_list} to {entry_type} {item_num}")
            
        except ValueError as e:
            app_state.log_message(f"ERROR: {str(e)}. Use: tag <item_number> tag1 tag2 [[multi word tag]] tag3", True)
    else:
        app_state.log_message(f"ERROR: Unknown command '{command}'", True)
