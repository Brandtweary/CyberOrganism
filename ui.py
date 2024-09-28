import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton
from PySide6.QtGui import QFont, QColor, QPalette, QCursor, QGuiApplication, QScreen, QPainter, Qt
from PySide6.QtCore import Qt, QSize
from drawing import SimAreaWidget
import time
from shared_resources import debug
from widgets import StatBlock, CollapsibleBox, FoldingStatBlock


class UI:
    def __init__(self):
        self.app = QApplication(sys.argv)
        
        font = QFont("Roboto Mono", 12)
        self.app.setFont(font)
        
        self.main_window = QMainWindow()
        self.should_exit = False
        self.loaded = False

        # Get the primary screen
        screen = QGuiApplication.primaryScreen()
        
        # Get the screen size
        self.WIDTH, self.HEIGHT = screen.size().width(), screen.size().height()
        self.FPS = 30

        # Set up UI elements
        self.setup_ui()

        # Set up event handlers
        self.main_window.contextMenuEvent = self.handle_context_menu
        self.main_window.closeEvent = self.handle_close_event

    def setup_ui(self):
        self.setup_main_window()
        self.setup_sim_area()
        self.setup_left_sidebar()
        self.setup_loading_screen()

        self.set_styles()
        self.setup_key_handlers()
        self.main_window.showFullScreen()
        
        # Initially hide all UI elements except the main window and loading screen
        self.hide_ui_elements()

    def setup_loading_screen(self):
        self.loading_screen = QWidget(self.main_window)
        self.loading_screen.resize(self.main_window.size())
        self.loading_screen.raise_()  # Ensure it's on top
        self.loading_screen.show()  # Initially shown

    def set_styles(self):
        # Set main window background color to black
        self.main_window.setStyleSheet("background-color: black;")

        # Set sim_area background color to black
        self.sim_area.setStyleSheet("background-color: black;")

        # Set sidebar background color to slate gray
        self.left_sidebar.setStyleSheet("""
            QWidget#leftSidebar {
                background-color: #323436;
            }
        """)

        # Set global style for the main window and its children
        self.main_window.setStyleSheet("""
            QWidget {
                color: #00FF00;  /* Green text */
            }
        """)

        # Set loading screen background color to black
        self.loading_screen.setStyleSheet("background-color: black;")

    def hide_ui_elements(self):
        self.left_sidebar.hide()
        self.sim_area.hide()
        self.loading_screen.show()

    def show_ui_elements(self):
        self.left_sidebar.show()
        self.sim_area.show()
        self.loading_screen.hide()

    def setup_main_window(self):
        self.main_window.setWindowTitle("CyberOrganism")
        central_widget = QWidget()
        self.main_window.setCentralWidget(central_widget)
        self.main_window.resize(self.WIDTH, self.HEIGHT)

    def setup_sim_area(self):
        self.sim_area = SimAreaWidget(self.FPS, self.main_window)
        self.sim_area.setGeometry(0, 0, self.WIDTH, self.HEIGHT)

    def setup_left_sidebar(self):
        # Create the actual sidebar widget
        self.left_sidebar = QWidget(self.main_window)
        self.left_sidebar.setObjectName("leftSidebar")
        self.left_sidebar.setGeometry(0, 0, 300, self.HEIGHT)  # Adjust for border

        # Create a layout for the sidebar and remove its margins
        sidebar_layout = QVBoxLayout()
        sidebar_layout.setContentsMargins(0, 0, 0, 0)  # Remove inner margins
        sidebar_layout.setSpacing(0)  # Remove spacing between widgets
        self.left_sidebar.setLayout(sidebar_layout)

        self.setup_stat_section("Organism Statistics", sidebar_layout, "organism_stats")
        self.setup_stat_section("Performance Statistics", sidebar_layout, "performance_stats")
        self.setup_stat_section("Training Statistics", sidebar_layout, "training_stats")
        if debug:
            self.setup_stat_section("Debug Info", sidebar_layout, "debug_info", expanded=False)  # Add this line

        sidebar_layout.addStretch(1)  # This pushes everything to the top

        # Ensure the sidebar is on top of the frame
        self.left_sidebar.raise_()

    def setup_stat_section(self, title, parent_layout, label_attr_name, expanded=True):
        stats_box = CollapsibleBox(title, self.left_sidebar)
        parent_layout.addWidget(stats_box)
        
        stats_layout = QVBoxLayout()
        stats_layout.setSpacing(3)
        stats_layout.setContentsMargins(5, 5, 5, 5)
        
        setattr(self, f"{label_attr_name}_layout", stats_layout)
        setattr(self, f"{label_attr_name}_blocks", {})
        
        if label_attr_name == "training_stats":
            self.action_distribution_widget = FoldingStatBlock("Action Distribution", self.left_sidebar)
            stats_layout.addWidget(self.action_distribution_widget)

        stats_box.setContentLayout(stats_layout)
        stats_box.setExpanded(expanded)

    def setup_key_handlers(self):
        self.main_window.keyPressEvent = self.handle_key_press

    def handle_key_press(self, event):
        if event.key() == Qt.Key_Escape:
            self.should_exit = True
            self.main_window.close()

    def handle_context_menu(self, event):
        if not self.left_sidebar.geometry().contains(event.pos()):
            self.toggle_sidebar()

    def toggle_sidebar(self):
        if self.left_sidebar.isVisible():
            self.left_sidebar.hide()
        else:
            self.left_sidebar.show()

    def handle_close_event(self, event):
        self.should_exit = True
        event.accept()

    def update_left_sidebar(self, organism_stats, performance_stats, training_stats, action_distribution):
        if organism_stats is not None:
            self.update_stat_section(organism_stats, "organism_stats")
        if performance_stats is not None:
            self.update_stat_section(performance_stats, "performance_stats")
        if training_stats is not None:
            self.update_stat_section(training_stats, "training_stats")
        if action_distribution is not None:
            self.action_distribution_widget.update_content(action_distribution)

    def update_stat_section(self, stats, section_name):
        layout = getattr(self, f"{section_name}_layout")
        blocks = getattr(self, f"{section_name}_blocks")

        # If it's the training stats section, temporarily remove the action distribution widget
        if section_name == "training_stats":
            layout.removeWidget(self.action_distribution_widget)

        for name, value in stats.items():
            if name not in blocks:
                block = StatBlock(name, value)
                layout.addWidget(block)
                blocks[name] = block
            else:
                blocks[name].update_value(value)
        
        for name in list(blocks.keys()):
            if name not in stats:
                blocks[name].deleteLater()
                del blocks[name]
        
        # Find the maximum width of name labels
        max_width = max(block.name_label.sizeHint().width() for block in blocks.values())

        # Set all name labels to the maximum width
        for block in blocks.values():
            block.set_name_width(max_width)

        # If it's the training stats section, add the action distribution widget back at the end
        if section_name == "training_stats":
            layout.addWidget(self.action_distribution_widget)

    def update_debug_info(self):
        if not debug:
            return
        
        mouse_x, mouse_y = self.sim_area.get_mouse_position()
        screen = QGuiApplication.primaryScreen()
        device_pixel_ratio = screen.devicePixelRatio()
        screen_size = screen.size()
        window_geometry = self.main_window.geometry()
        
        debug_info = {
            "Mouse Position": f"({mouse_x}, {mouse_y})",
            "Device Pixel Ratio": f"{device_pixel_ratio}",
            "Screen Size": f"{screen_size.width()}x{screen_size.height()}",
            "Window Geometry": f"{window_geometry.width()}x{window_geometry.height()} at ({window_geometry.x()}, {window_geometry.y()})"
        }
        
        self.update_stat_section(debug_info, "debug_info")

    def update(self, sim_state):
        if sim_state.frame_count <= sim_state.loading_frames:
            # Keep UI elements hidden during loading
            self.hide_ui_elements()
        elif not self.loaded:
            # Show UI elements after loading is complete
            self.show_ui_elements()
            self.loaded = True

        self.sim_area.draw_simulation(sim_state)
        self.update_debug_info()
