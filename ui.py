import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton
from PySide6.QtGui import QFont, QColor, QPalette, QCursor, QGuiApplication, QScreen, QPainter, Qt
from PySide6.QtCore import Qt, QSize
from drawing import SimAreaWidget
import time
from shared_resources import debug

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

        self.setup_stat_section("Organism Statistics", sidebar_layout, "organism_stats_label")
        self.setup_stat_section("Performance Statistics", sidebar_layout, "performance_stats_label")
        self.setup_stat_section("Training Metrics", sidebar_layout, "training_metrics_label")
        if debug:
            self.setup_stat_section("Debug Info", sidebar_layout, "debug_info_label", expanded=False)  # Add this line

        sidebar_layout.addStretch(1)  # This pushes everything to the top

        # Ensure the sidebar is on top of the frame
        self.left_sidebar.raise_()

    def setup_stat_section(self, title, parent_layout, label_attr_name, expanded=True):
        stats_box = CollapsibleBox(title, self.left_sidebar)
        parent_layout.addWidget(stats_box)
        
        stats_layout = QVBoxLayout()
        stats_label = QLabel("", self.left_sidebar)
        stats_label.setWordWrap(True)  # Allow text to wrap
        stats_layout.addWidget(stats_label)
        
        stats_box.setContentLayout(stats_layout)
        stats_box.setExpanded(expanded)
        setattr(self, label_attr_name, stats_label)

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

    def update_left_sidebar(self, organism_stats, performance_stats, training_metrics):
        if organism_stats is not None:
            self.organism_stats_label.setText(self.format_stats(organism_stats))
        if performance_stats is not None:
            self.performance_stats_label.setText(self.format_stats(performance_stats))
        if training_metrics is not None:
            self.training_metrics_label.setText(self.format_training_metrics(training_metrics))

    def format_stats(self, stats):
        return "\n".join(stats)

    def format_training_metrics(self, training_metrics):
        formatted_metrics = []
        combined = training_metrics["combined_averages"]
        formatted_metrics.extend([
            f"Combined Loss: {combined['loss_window_avg']:.4f}",
            f"Combined Q-Value: {combined['current_q_window_avg']:.4f}",
            f"Combined Expected Q: {combined['expected_q_window_avg']:.4f}",
            f"Reward: {training_metrics['reward']['reward_window_avg']:.4f}"
        ])
        for action, metrics in training_metrics.items():
            if action not in ["combined_averages", "reward"]:
                formatted_metrics.extend([
                    f"Action {action} Loss: {metrics['loss_window_avg']:.4f}",
                    f"Action {action} Q-Value: {metrics['current_q_window_avg']:.4f}",
                    f"Action {action} Expected Q: {metrics['expected_q_window_avg']:.4f}"
                ])
        return "\n".join(formatted_metrics)

    def update_debug_info(self):
        if not debug:
            return
        
        sim_area_width, sim_area_height = self.sim_area.width(), self.sim_area.height()
        mouse_x, mouse_y = self.sim_area.get_mouse_position()
        
        # Get the primary screen
        screen = QGuiApplication.primaryScreen()
        
        # Get the device pixel ratio
        device_pixel_ratio = screen.devicePixelRatio()
        
        # Get the true screen size
        screen_size = screen.size()
        
        # Get the main window geometry
        window_geometry = self.main_window.geometry()
        
        debug_info = f"Sim Area: {sim_area_width}x{sim_area_height}\n"
        debug_info += f"Mouse Position: ({mouse_x}, {mouse_y})\n"
        debug_info += f"Device Pixel Ratio: {device_pixel_ratio}\n"
        debug_info += f"Screen Size: {screen_size.width()}x{screen_size.height()}\n"
        debug_info += f"Window Geometry: {window_geometry.width()}x{window_geometry.height()} at ({window_geometry.x()}, {window_geometry.y()})\n"
        debug_info += f"Logical DPI: {screen.logicalDotsPerInch()}\n"
        debug_info += f"Physical DPI: {screen.physicalDotsPerInch()}"
        
        self.debug_info_label.setText(debug_info)

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

class CollapsibleBox(QWidget):
    def __init__(self, title="", parent=None):
        super(CollapsibleBox, self).__init__(parent)

        self.toggle_button = QPushButton(title)
        
        # Set font explicitly
        font = QFont("Roboto Mono", 14)  # Adjust size as needed
        self.toggle_button.setFont(font)
        
        self.toggle_button.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding: 5px;
                background-color: #1A1A1A;  /* Dark gray, nearly black */
                border: none;
                color: #00FF00;  /* Ensure the button text is green */
            }
            QPushButton:hover {
                background-color: #252525;  /* Slightly lighter */
            }
        """)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(False)

        self.content_area = QWidget()
        self.content_area.setVisible(False)

        lay = QVBoxLayout(self)
        lay.setSpacing(0)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.toggle_button)
        lay.addWidget(self.content_area)

        self.main_layout = QVBoxLayout()
        self.content_area.setLayout(self.main_layout)

        self.toggle_button.toggled.connect(self.on_toggle)  # Connect the signal

    def on_toggle(self, checked):
        self.content_area.setVisible(checked)
        self.adjustSize()
        self.updateGeometry()

    def setContentLayout(self, layout):
        # Remove all items from the main_layout
        while self.main_layout.count():
            item = self.main_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Add the new layout to main_layout
        self.main_layout.addLayout(layout)

    def setExpanded(self, expanded):
        self.toggle_button.setChecked(expanded)
        self.content_area.setVisible(expanded)
