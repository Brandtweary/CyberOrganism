from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QPushButton, QVBoxLayout
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt, Signal


class StatBlock(QWidget):
    def __init__(self, name, value, parent=None):
        super(StatBlock, self).__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.name_label = QLabel(name)
        self.value_label = QLabel(str(value))
        self.name_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.value_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.addWidget(self.name_label)
        layout.addWidget(self.value_label)

    def update_value(self, value):
        self.value_label.setText(str(value))
    
    def set_name_width(self, width):
        self.name_label.setMinimumWidth(width)


class FoldingStatBlock(QWidget):
    toggled = Signal(bool)

    def __init__(self, name, parent=None):
        super(FoldingStatBlock, self).__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        self.title = str(name)
        self.toggle_button = QPushButton()
        self.toggle_button.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding: 5px;
                background-color: transparent;
                border: none;
                color: #00FF00;
            }
            QPushButton:hover {
                background-color: #252525;
            }
        """)
        self.toggle_button.setCheckable(True)
        self.toggle_button.toggled.connect(self.on_toggle)

        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(5, 3, 5, 3)  # Reduced top and bottom margins
        self.content_layout.setSpacing(2)  # Tighter spacing between items
        self.content_widget.hide()

        self.layout.addWidget(self.toggle_button)
        self.layout.addWidget(self.content_widget)

        self.stat_blocks = {}  # Map to store StatBlock widgets
        self.update_arrow()

    def on_toggle(self, checked):
        self.content_widget.setVisible(checked)
        self.update_arrow()
        self.toggled.emit(checked)

    def update_arrow(self):
        arrow = "▼" if self.toggle_button.isChecked() else "▶"
        self.toggle_button.setText(f"{arrow} {self.title}")

    def update_content(self, data):
        for name, value in data.items():
            if name not in self.stat_blocks:
                stat_block = StatBlock(name, self.format_value(value))
                self.content_layout.addWidget(stat_block)
                self.stat_blocks[name] = stat_block
            else:
                self.stat_blocks[name].update_value(self.format_value(value))
        
        # Remove any stat blocks that are no longer in the data
        for name in list(self.stat_blocks.keys()):
            if name not in data:
                self.stat_blocks[name].deleteLater()
                del self.stat_blocks[name]
        
        # Adjust name widths
        if self.stat_blocks:
            max_width = max(block.name_label.sizeHint().width() for block in self.stat_blocks.values())
            for block in self.stat_blocks.values():
                block.set_name_width(max_width)
        
        # Adjust layout if necessary
        self.content_widget.adjustSize()
        self.adjustSize()

    def format_value(self, value):
        if isinstance(value, float):
            return f"{value:.2%}"
        return str(value)


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
