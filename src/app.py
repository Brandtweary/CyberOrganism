from prompt_toolkit import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.filters import Condition

from .store import TaskStore
from .models import Task, Note, Tag
from .ui import create_style, create_layout
from .commands import COMMANDS, process_hashtags
from .keybindings import create_keybindings

class InputMode:
    NOTE = "NOTE"
    COMMAND = "COMMAND"
    TASK = "TASK"
    EDIT = "EDIT"
    COMPLETE = "COMPLETE"

class CyberOrganism:
    def __init__(self, store=None):
        self.store = store if store is not None else TaskStore()
        self.mode = InputMode.EDIT  # Start in edit mode
        self.command_mode = False
        self.running = True
        self.status_message = None  # Store current status/warning message
        self.message_style = 'status'  # Can be 'status' or 'warning'
        self.selected_index = 0  # Track currently selected item
        self.seen_commands = False  # Track if user has seen command menu
        self.InputMode = InputMode  # Make InputMode accessible to other modules
        
        # Create input buffer with proper configuration
        self.input_buffer = Buffer(
            multiline=True,  # Enable multi-line input
            enable_history_search=True  # Enable up/down history search
        )
        
        # Create UI components
        self.style = create_style()
        
        # Create layout
        self.layout = create_layout(
            self.get_taskpad_content,
            self.get_commands_content,
            self.input_buffer,
            self.get_prompt,
            self.get_help_message,
            lambda: self.command_mode,  # Pass as lambda to make it dynamic
            lambda: self.seen_commands  # Pass as lambda to make it dynamic
        )
        
        # Setup key bindings
        self.kb = create_keybindings(self)
        
        # Create application
        self.app = Application(
            layout=self.layout,
            key_bindings=self.kb,
            style=self.style,
            full_screen=False,
            erase_when_done=False,
            mouse_support=False
        )
        self.kb_listener = None  # Store reference to keyboard listener

    def get_taskpad_content(self):
        """Generate the main content area text"""
        lines = [
            ('class:title', "CYBERORGANISM TASKPAD\n\n"),
            ('class:header', "ID    TYPE     CONTENT\n"),
            ('class:header', "---   ----     -------\n")
        ]
        
        def format_tag(tag_name: str) -> str:
            # If tag contains spaces, use [[tag name]] format
            if ' ' in tag_name:
                return f"#[[{tag_name}]]"
            return f"#{tag_name}"

        # Get all entries to display
        display_entries = []
        
        # In edit mode, add the new task slot as the first entry
        if self.mode == InputMode.EDIT:
            display_entries.append(("NEW", "[Type here to create new tasks]", []))
        
        # Add all tasks
        for task in self.store.get_all_tasks():
            display_entries.append(("TASK", task.title, task.tags))
            
        # Add all notes
        for note in self.store.get_all_notes():
            display_entries.append(("NOTE", note.content[:50] + "..." if len(note.content) > 50 else note.content, note.tags))
        
        # Display all entries with their actual index
        for i, (entry_type, content, tags) in enumerate(display_entries):
            tags_str = " " + " ".join(format_tag(t.name) for t in tags) if tags else ""
            style = 'class:content reverse' if self.mode == InputMode.EDIT and i == self.selected_index else 'class:content'
            lines.extend([
                (style, f"{i:<6}"),
                (style, f"{entry_type:<8}"),
                (style, f"{content}{tags_str}\n")
            ])
            
        # Add status message if present
        if self.status_message:
            lines.extend([
                ('class:content', "\n"),  # Empty line before status
                (f'class:{self.message_style}', f"{self.status_message}\n")
            ])
            
        return lines

    def get_commands_content(self):
        """Generate the commands area text"""
        if not self.command_mode:
            return []
            
        lines = [
            ('class:title', "\nAVAILABLE COMMANDS\n\n")
        ]
        
        for cmd, desc in COMMANDS.items():
            lines.extend([
                ('class:command', f"{cmd:<10}"),
                ('class:content', f"{desc}\n")
            ])
            
        lines.append(('class:content', "\n"))  # Add extra newline at the end
        return lines

    def get_prompt(self):
        """Generate the input prompt"""
        mode_str = "COMMAND" if self.command_mode else self.mode
        prompt_str = "> "
        return FormattedText([
            ('class:mode', mode_str),
            ('class:prompt', f" {prompt_str}")
        ])

    def get_help_message(self):
        """Generate the help message"""
        # Only show help if user hasn't entered any commands or used command mode
        if self.seen_commands:
            return []
        return FormattedText([
            ('class:help', "Press 'Escape' to enter command mode, or type commands directly using '/' such as '/task', '/exit', etc.")
        ])

    def log_message(self, message: str, style: str = 'status'):
        """Set a status message with optional style"""
        self.status_message = message
        self.message_style = style
        self.app.invalidate()

    def _prefill_selected_content(self):
        """Pre-fill input buffer with currently selected item's content"""
        # Get all entries in display order
        entries = []
        if self.mode == InputMode.EDIT:
            entries.append(None)  # Placeholder for new task slot
        entries.extend(self.store.get_all_tasks())
        entries.extend(self.store.get_all_notes())
        
        # Get the selected entry
        if self.selected_index >= len(entries):
            return
            
        entry = entries[self.selected_index]
        
        # Handle new task slot
        if entry is None:
            self.input_buffer.text = ""
            self.input_buffer.cursor_position = 0
            return
            
        # Handle tasks and notes
        if isinstance(entry, Task):
            content = entry.title
        else:  # Note
            content = entry.content
            
        # Add tags if present
        if entry.tags:
            tag_str = " " + " ".join(
                f"#[[{t.name}]]" if ' ' in t.name else f"#{t.name}" 
                for t in entry.tags
            )
            content += tag_str
            
        self.input_buffer.text = content
        self.input_buffer.cursor_position = len(content)

    def handle_command(self, command: str) -> bool:
        """Handle a command string"""
        cmd = command.lower().strip()
        self.status_message = None  # Clear previous message
        
        if cmd == "exit":
            self.app.exit()
            return False
        elif cmd == "task":
            self.mode = InputMode.TASK
            self.command_mode = False
            self.log_message("Switched to TASK mode")
        elif cmd == "note":
            self.mode = InputMode.NOTE
            self.command_mode = False
            self.log_message("Switched to NOTE mode")
        elif cmd == "edit":
            self.mode = InputMode.EDIT
            self.command_mode = False
            # Always start with the "new task" slot selected
            self.selected_index = 0
            self._prefill_selected_content()
            self.app.invalidate()
            self.log_message(f"Switched to EDIT mode")
        elif cmd == "complete" or command.startswith("complete "):
            if command.startswith("complete "):
                # Handle direct task completion
                try:
                    task_num = int(command.split(" ", 1)[1].strip())
                    # Get entries in display order
                    display_entries = []
                    if self.mode == InputMode.EDIT:
                        display_entries.append(None)  # Placeholder for new task slot
                    display_entries.extend(self.store.get_all_tasks())
                    display_entries.extend(self.store.get_all_notes())

                    if task_num >= len(display_entries) or task_num < 0:
                        self.log_message("Invalid task number", 'warning')
                    elif not isinstance(display_entries[task_num], Task):
                        self.log_message("Cannot complete a note or new task slot", 'warning')
                    else:
                        task = display_entries[task_num]
                        task.complete()
                        self.store._save()
                        self.log_message(f"Marked task {task_num} as completed")
                except ValueError:
                    self.log_message("Invalid task number", 'warning')
            else:
                # Enter complete mode
                self.mode = InputMode.COMPLETE
                self.command_mode = False
                self.log_message("Switched to COMPLETE mode - Enter task number to complete")
        elif cmd == "command":  # Hidden command to enter command mode
            self.command_mode = True
            self.app.invalidate()
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
                if self.mode == InputMode.EDIT:
                    entries.append(None)  # Placeholder for new task slot
                entries.extend(self.store.get_all_tasks())
                entries.extend(self.store.get_all_notes())
                
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
                self.log_message(f"Added tags {tag_list} to {entry_type} {item_num}")
                
            except ValueError as e:
                self.log_message(f"ERROR: {str(e)}. Use: tag <item_number> tag1 tag2 [[multi word tag]] tag3", 'warning')
        else:
            self.log_message(f"ERROR: Unknown command '{command}'", 'warning')
        return True

    def run(self):
        """Run the application"""
        try:
            self.app.run()
        finally:
            if self.kb_listener:
                self.kb_listener.stop()
