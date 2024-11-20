import asyncio
import cmd
from shared_resources import DATA_DIR, logger, token_logger
from documents import create_data_snapshot, process_documents
from tests import test_api_queue


# ANSI color codes
GREEN = "\033[32m"
BLUE = "\033[34m"
RESET = "\033[0m"

class CymbiontShell(cmd.Cmd):
    intro = f'{GREEN}Welcome to Cymbiont. Type help or ? to list commands.{RESET}\n'
    prompt = f'{BLUE}cymbiont>{RESET} '
    
    def __init__(self, loop: asyncio.AbstractEventLoop):
        super().__init__()
        self.loop: asyncio.AbstractEventLoop = loop
    
    def print_topics(self, header: str, cmds: list[str] | None, cmdlen: int, maxcol: int) -> None:
        """Override to add color to help topics"""
        if not cmds:  # Skip empty sections
            return
        if header:
            if header == "Documented commands (type help <topic>):":
                header = f"{GREEN}Available commands (type help <command>):{RESET}"
            self.stdout.write(f"{GREEN}{header}{RESET}\n")
        self.columnize([f"{GREEN}{cmd}{RESET}" for cmd in cmds], maxcol-1)
        self.stdout.write("\n")

    def print_help_text(self, text: str) -> None:
        """Helper method to print help text in green"""
        self.stdout.write(f"{GREEN}{text}{RESET}\n")

    def do_help(self, arg: str) -> None:
        """Override the help command to add color"""
        if arg:
            # Show help for specific command
            try:
                func = getattr(self, 'help_' + arg)
            except AttributeError:
                try:
                    doc = getattr(self, 'do_' + arg).__doc__
                    if doc:
                        self.print_help_text(str(doc))
                        return
                except AttributeError:
                    pass
                self.print_help_text(f"*** No help on {arg}")
        else:
            # Show the list of commands
            super().do_help(arg)
    
    def do_process_documents(self, arg: str) -> None:
        """Process documents in the data directory.
        Usage: process_documents"""
        try:
            token_logger.reset_tokens()
            future = asyncio.run_coroutine_threadsafe(
                process_documents(DATA_DIR),
                self.loop
            )
            future.result()
            token_logger.print_tokens()
        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
    
    def do_exit(self, arg: str) -> bool:
        """Exit the Cymbiont shell"""
        self.loop.call_soon_threadsafe(self.loop.stop)
        return True
    
    def do_test_api_queue(self, arg: str) -> None:
        """Run API queue tests.
        Usage: test_api_queue"""
        try:
            future = asyncio.run_coroutine_threadsafe(
                test_api_queue.run_tests(),
                self.loop
            )
            future.result()  # Adjust timeout as needed
        except Exception as e:
            logger.error(f"API queue tests failed: {str(e)}")
    
    def do_create_data_snapshot(self, arg: str) -> None:
        """Create a snapshot of the data directory structure."""
        if not arg:
            print("Error: Please provide a name for the snapshot")
            return
        
        try:
            token_logger.reset_tokens()
            future = asyncio.run_coroutine_threadsafe(
                create_data_snapshot(arg),
                self.loop
            )
            snapshot_path = future.result()
            logger.info(f"Created snapshot at {snapshot_path}")
            token_logger.print_tokens()
        except Exception as e:
            logger.error(f"Snapshot creation failed: {str(e)}")