#!/usr/bin/env python3
from pynput import keyboard
from prompt_toolkit import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.layout.controls import BufferControl

class TestApp:
    def __init__(self):
        # Create input buffer
        self.input_buffer = Buffer(multiline=True)
        
        # Create key bindings
        self.kb = KeyBindings()
        
        # Create keyboard controller for checking ctrl state
        self.keyboard = keyboard.Controller()
        
        # Setup UI
        self.layout = Layout(
            HSplit([
                Window(BufferControl(buffer=self.input_buffer))
            ])
        )
        
        # Create application
        self.app = Application(
            layout=self.layout,
            key_bindings=self.kb,
            full_screen=False,
            erase_when_done=False
        )
        
        # Add enter binding - no filter this time
        @self.kb.add('enter')
        def handle_enter(event):
            """Process input when enter is pressed."""
            text = self.input_buffer.text
            
            # Check if ctrl is currently being held down
            if self.keyboard.pressed(keyboard.Key.ctrl):
                print(f"Ctrl+Enter detected! Text: {text}")
                # Only clear buffer on ctrl+enter
                self.input_buffer.reset()
            else:
                # For regular enter, add the newline to the buffer
                self.input_buffer.insert_text('\n')
    
    def run(self):
        print("Starting test app...")
        print("Type some text and press:")
        print("- Enter: Add newline")
        print("- Ctrl+Enter: Process text")
        print("- Ctrl+C: Exit")
        
        self.app.run()

if __name__ == '__main__':
    app = TestApp()
    app.run()
