import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QRubberBand
from PySide6.QtGui import QFont, QColor, QPalette, QCursor, QGuiApplication, QScreen, QPainter, Qt
from PySide6.QtCore import Qt, QSize, QTimer, QRect, QPoint
from drawing import SimAreaWidget
import time
from widgets import StatBlock, CollapsibleBox, FoldingStatBlock

debug = False

class UI:
    def __init__(self):
        self.app = QApplication(sys.argv)
        
        font = QFont("Roboto Mono", 12)
        self.app.setFont(font)
        
        self.main_window = QMainWindow()
        self.should_exit = False
        self.FPS = 30
        self.loaded = False
        self.loading_frames = 2 * self.FPS  # 2 seconds of loading in ideal case

        # Get the primary screen
        screen = QGuiApplication.primaryScreen()
        
        # Get the screen size
        self.WIDTH, self.HEIGHT = screen.size().width(), screen.size().height()
        
        # Set up UI elements
        self.setup_ui()

        # Set up event handlers
        self.main_window.closeEvent = self.handle_close_event

        self.last_right_click_time = 0
        self.sidebar_toggle_timer = QTimer()
        self.sidebar_toggle_timer.setSingleShot(True)
        self.sidebar_toggle_timer.timeout.connect(self.toggle_sidebar)

        self.rubber_band = None
        self.origin = QPoint()

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
        self.setup_stat_section("Simulation Statistics", sidebar_layout, "simulation_stats")
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
        self.main_window.mousePressEvent = self.handle_mouse_press
        self.main_window.mouseMoveEvent = self.handle_mouse_move
        self.main_window.mouseReleaseEvent = self.handle_mouse_release

    def handle_key_press(self, event):
        if event.key() == Qt.Key_Escape:
            self.should_exit = True
            self.main_window.close()

    def handle_mouse_press(self, event):
        if event.button() == Qt.LeftButton:
            self.origin = event.pos()
            if not self.rubber_band:
                self.rubber_band = QRubberBand(QRubberBand.Rectangle, self.main_window)
            self.rubber_band.setGeometry(QRect(self.origin, QSize()))
            self.rubber_band.show()
        elif event.button() == Qt.RightButton:
            current_time = time.time()
            if current_time - self.last_right_click_time < 0.2:
                self.sidebar_toggle_timer.stop()
                self.center_viewport()
            else:
                self.sidebar_toggle_timer.start(200)
            self.last_right_click_time = current_time

    def handle_mouse_move(self, event):
        if self.rubber_band and not self.rubber_band.isHidden():
            self.rubber_band.setGeometry(QRect(self.origin, event.pos()).normalized())

    def handle_mouse_release(self, event):
        if event.button() == Qt.LeftButton and self.rubber_band:
            self.rubber_band.hide()
            selection_rect = self.rubber_band.geometry()
            self.check_organisms_in_selection(selection_rect)

    def check_organisms_in_selection(self, selection_rect):
        selected_organisms = []
        for organism in self.sim_area.sim_engine.organisms:
            screen_x, screen_y = self.sim_area.grid_to_screen(organism.x, organism.y)
            organism_pos = QPoint(screen_x, screen_y)
            if selection_rect.contains(organism_pos):
                selected_organisms.append(organism)

        if selected_organisms:
            new_test_organism = selected_organisms[0]
            if new_test_organism != self.sim_area.sim_engine.test_organism:
                self.sim_area.sim_engine.set_test_organism(new_test_organism)

    def toggle_sidebar(self):
        if self.left_sidebar.isVisible():
            self.left_sidebar.hide()
        else:
            self.left_sidebar.show()

    def handle_close_event(self, event):
        self.should_exit = True
        event.accept()

    def update_left_sidebar(self, organism_stats, performance_stats, training_stats, action_distribution, simulation_stats):
        if organism_stats is not None and organism_stats:
            self.update_stat_section(organism_stats, "organism_stats")
        if performance_stats is not None and performance_stats:
            self.update_stat_section(performance_stats, "performance_stats")
        if training_stats is not None and training_stats:
            self.update_stat_section(training_stats, "training_stats")
        if simulation_stats is not None and simulation_stats:
            self.update_stat_section(simulation_stats, "simulation_stats")
        if action_distribution is not None and action_distribution:
            self.action_distribution_widget.update_content(action_distribution)

    def update_stat_section(self, stats, section_name):
        layout = getattr(self, f"{section_name}_layout")
        blocks = getattr(self, f"{section_name}_blocks")

        if section_name == "training_stats":
            layout.removeWidget(self.action_distribution_widget)

        for name, value in stats:
            if name not in blocks:
                block = StatBlock(name, value)
                layout.addWidget(block)
                blocks[name] = block
            else:
                blocks[name].update_value(value)
        
        for name in list(blocks.keys()):
            if name not in [stat[0] for stat in stats]:
                blocks[name].deleteLater()
                del blocks[name]
        
        max_width = max(block.name_label.sizeHint().width() for block in blocks.values())
        for block in blocks.values():
            block.set_name_width(max_width)

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
        if sim_state.current_state.frame_count <= self.loading_frames:
            # Keep UI elements hidden during loading
            self.hide_ui_elements()
        elif not self.loaded:
            # Show UI elements after loading is complete
            self.show_ui_elements()
            self.loaded = True

        self.sim_area.draw_simulation(sim_state)
        self.update_debug_info()

        if self.rubber_band and not self.rubber_band.isHidden():
            self.rubber_band.update()

    def center_viewport(self):
        self.sim_area.center_viewport()
