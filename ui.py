import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QSizePolicy, QStackedLayout
from PySide6.QtGui import QFont, QColor, QPalette, QCursor
from PySide6.QtCore import Qt, QSize
from drawing import draw_simulation  # Assuming this is your drawing function
from screeninfo import get_monitors

class UI:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.main_window = QMainWindow()
        self.should_exit = False

        # Get monitor information
        self.monitor = get_monitors()[0]
        self.WIDTH, self.HEIGHT = self.monitor.width, self.monitor.height

        # Set up UI elements
        self.setup_ui()

    def setup_ui(self):
        self.setup_main_window()
        self.setup_sim_area()
        self.setup_left_sidebar()

        self.set_styles()
        self.setup_key_handlers()
        self.main_window.show()

    def setup_main_window(self):
        self.main_window.setWindowTitle("CyberOrganism")
        central_widget = QWidget()
        self.main_window.setCentralWidget(central_widget)
        self.main_window.resize(self.WIDTH, self.HEIGHT)

    def setup_sim_area(self):
        self.sim_area = QWidget(self.main_window)
        self.sim_area.setGeometry(0, 0, self.WIDTH, self.HEIGHT)
        
        # Set background color directly
        self.sim_area.setStyleSheet("background-color: red;")


    def setup_left_sidebar(self):
        self.left_sidebar = QWidget(self.main_window)
        self.left_sidebar.setGeometry(0, 0, 350, self.HEIGHT)

        sidebar_layout = QVBoxLayout(self.left_sidebar)

        self.setup_stat_section("Organism Statistics", sidebar_layout, "organism_stats_label")
        self.setup_stat_section("Performance Statistics", sidebar_layout, "performance_stats_label")
        self.setup_stat_section("Training Metrics", sidebar_layout, "training_metrics_label")

        self.left_sidebar.raise_()

    def setup_stat_section(self, title, parent_layout, label_attr_name):
        stats_box = CollapsibleBox(title, self.left_sidebar)
        parent_layout.addWidget(stats_box)
        stats_layout = QVBoxLayout()
        stats_label = QLabel("", self.left_sidebar)
        stats_layout.addWidget(stats_label)
        stats_box.setContentLayout(stats_layout)
        stats_box.setExpanded(True)
        setattr(self, label_attr_name, stats_label)

    def set_styles(self):
        # Set sim_area background color to red
        self.sim_area.setStyleSheet("background-color: red;")

        # Set sidebar background color to slate gray
        self.left_sidebar.setStyleSheet("background-color: #708090;")  # Hex code for slate gray

        # Set global font and text color
        font = QFont("Roboto Mono", 22)
        self.main_window.setFont(font)

        palette = self.main_window.palette()
        palette.setColor(QPalette.WindowText, QColor(0, 255, 0))  # Green text
        self.main_window.setPalette(palette)

    def setup_key_handlers(self):
        self.main_window.keyPressEvent = self.handle_key_press

    def handle_key_press(self, event):
        if event.key() == Qt.Key_Escape:
            self.should_exit = True
            self.main_window.close()

    def update_left_sidebar(self, organism_stats, performance_stats, training_metrics):
        self.organism_stats_label.setText(self.format_stats(organism_stats))
        self.performance_stats_label.setText(self.format_stats(performance_stats))
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

    def update(self, sim_state):
        #draw_simulation(self.sim_area, sim_state)  # Assuming this is your drawing function
        self.app.processEvents()

    def get_display_framerate(self):
        # You'll need to implement this if you want to get the framerate
        return 999

    def get_mouse_position(self):
        global_pos = QCursor.pos()
        local_pos = self.sim_area.mapFromGlobal(global_pos)
        return local_pos.x(), local_pos.y()

    def get_viewport_dimensions(self):  # rename to get_sim_area_dimensions
        return self.sim_area.width(), self.sim_area.height()

    def run(self):
        self.main_window.show()
        return self.app.exec()

class CollapsibleBox(QWidget):
    def __init__(self, title="", parent=None):
        super(CollapsibleBox, self).__init__(parent)

        self.toggle_button = QPushButton(title)
        self.toggle_button.setStyleSheet("text-align: left; padding: 5px;")
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(False)

        self.toggle_button.clicked.connect(self.on_toggle)

        self.content_area = QWidget()
        self.content_area.setVisible(False)

        lay = QVBoxLayout(self)
        lay.setSpacing(0)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.toggle_button)
        lay.addWidget(self.content_area)

        self.main_layout = QVBoxLayout(self.content_area)

    def on_toggle(self, checked):
        self.content_area.setVisible(checked)

    def setContentLayout(self, layout):
        lay = self.main_layout
        del lay
        self.main_layout = layout
        self.content_area.setLayout(self.main_layout)

    def setExpanded(self, expanded):
        self.toggle_button.setChecked(expanded)
        self.content_area.setVisible(expanded)

