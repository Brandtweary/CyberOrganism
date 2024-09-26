from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QPushButton, QVBoxLayout
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt


class StatBlock(QWidget):
    def __init__(self, name, value, parent=None):
        super(StatBlock, self).__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.name_label = QLabel(name)
        self.value_label = QLabel(value)
        self.name_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.value_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.addWidget(self.name_label)
        layout.addWidget(self.value_label)

    def update_value(self, value):
        self.value_label.setText(value)
    
    def set_name_width(self, width):
        self.name_label.setMinimumWidth(width)


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
