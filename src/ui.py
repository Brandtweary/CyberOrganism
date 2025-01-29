from prompt_toolkit.layout import Layout, Window, HSplit, FormattedTextControl, Dimension
from prompt_toolkit.layout.containers import WindowAlign
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.layout.processors import BeforeInput
from prompt_toolkit.layout.controls import BufferControl
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.styles import Style as PromptStyle

# Define colors as hex codes
NEON_GREEN = '#57ff14'
NEON_CYAN = '#00ffff'

def create_style():
    """Create the application style"""
    return PromptStyle.from_dict({
        'title': f'{NEON_GREEN} bold',  # Green bold for title
        'header': NEON_GREEN,  # Green for headers
        'content': NEON_GREEN,  # Green for content
        'mode': f'{NEON_GREEN} bold',  # Green bold for mode
        'prompt': NEON_GREEN,  # Green for prompt
        'command': NEON_CYAN,  # Cyan for commands
        'help': '#888888 italic',  # Help text in gray and italic
        'status': NEON_GREEN,  # Green for status messages
        'error': '#ff0000',  # Red for error messages
        'warning': '#ff0000 bold',  # Red bold for warnings
        'success': NEON_GREEN,  # Green for success messages
    })

def create_layout(taskpad_content_fn, commands_content_fn, input_buffer, get_prompt_fn, get_help_message_fn, command_mode_fn, seen_commands_fn):
    """Create the main application layout"""
    taskpad = FormattedTextControl(taskpad_content_fn)
    commands = FormattedTextControl(commands_content_fn)
    
    return Layout(
        HSplit([
            # Main taskpad area
            Window(
                content=taskpad,
                wrap_lines=True,
                height=Dimension(preferred=15)
            ),
            # Command area
            Window(
                content=commands,
                height=lambda: 8 if command_mode_fn() else 0
            ),
            # Help message
            Window(
                content=FormattedTextControl(get_help_message_fn),
                height=lambda: 1 if not seen_commands_fn() else 0
            ),
            # Empty line for spacing in command mode
            Window(height=lambda: 1 if command_mode_fn() else 0),
            # Input prompt at bottom
            Window(
                BufferControl(
                    buffer=input_buffer,
                    input_processors=[BeforeInput(get_prompt_fn)],
                    focusable=True
                ),
                height=5,  # Give more vertical space for multi-line input
                align=WindowAlign.LEFT,
                wrap_lines=True,
                always_hide_cursor=False
            )
        ])
    )
