from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QPainter, QColor, QCursor
from PySide6.QtCore import QRectF, Qt
from state_snapshot import ObjectType
import time
from shared_resources import debug

class SimAreaWidget(QWidget):
    def __init__(self, FPS, parent=None):
        super().__init__(parent)
        self.sim_state = None
        self.sim_engine = None
        self.FPS = FPS  # Frames per second
        self.camera_pan_threshold = 5  # Pixels
        self.camera_pan_speed = 150  # Pixels per frame

         # Color constants
        self.BLACK = (0, 0, 0)
        self.NEON_GREEN = (57, 255, 20)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.YELLOW = (255, 255, 0)
        self.PURPLE = (128, 0, 128)
        self.ORANGE_HALO = (255, 165, 0, 100)  # Including alpha value

    def draw_simulation(self, sim_state):
        self.sim_state = sim_state
        self.handle_camera_panning()
        self.update()  # Schedule a repaint

    def handle_camera_panning(self):
        mouse_x, mouse_y = self.get_mouse_position()
        sim_area_width, sim_area_height = self.width(), self.height()
        dx = dy = 0

        pan_amount = self.camera_pan_speed / self.FPS

        if mouse_x <= self.camera_pan_threshold:
            dx = -pan_amount
        elif mouse_x >= sim_area_width - self.camera_pan_threshold:
            dx = pan_amount

        if mouse_y <= self.camera_pan_threshold:
            dy = -pan_amount
        elif mouse_y >= sim_area_height - self.camera_pan_threshold:
            dy = pan_amount

        if dx != 0 or dy != 0:
            self.update_viewport(int(dx), int(dy))

    def update_viewport(self, dx=0, dy=0):
        self.update_viewport_dimensions()

        self.sim_engine.viewport_cell_center_x = max(
            self.sim_engine.viewport_cell_width // 2,
            min(
                self.sim_engine.viewport_cell_center_x + dx,
                self.sim_engine.world_width - self.sim_engine.viewport_cell_width // 2
            )
        )
        self.sim_engine.viewport_cell_center_y = max(
            self.sim_engine.viewport_cell_height // 2,
            min(
                self.sim_engine.viewport_cell_center_y + dy,
                self.sim_engine.world_height - self.sim_engine.viewport_cell_height // 2
            )
        )

    def update_viewport_dimensions(self):
        sim_area_width, sim_area_height = self.width(), self.height()
        self.sim_engine.viewport_cell_width = sim_area_width // self.sim_engine.CELL_SIZE
        self.sim_engine.viewport_cell_height = sim_area_height // self.sim_engine.CELL_SIZE

    def grid_to_screen(self, grid_x: int, grid_y: int):
        viewport_cell_x = grid_x - (self.sim_engine.viewport_cell_center_x - self.sim_engine.viewport_cell_width // 2)
        viewport_cell_y = grid_y - (self.sim_engine.viewport_cell_center_y - self.sim_engine.viewport_cell_height // 2)

        screen_x = viewport_cell_x * self.sim_engine.CELL_SIZE
        screen_y = viewport_cell_y * self.sim_engine.CELL_SIZE

        return int(screen_x), int(screen_y)

    def get_mouse_position(self):
        return QCursor.pos().x(), QCursor.pos().y()

    def paintEvent(self, event):
        if self.sim_state is None:
            return
        
        painter = QPainter(self)
        self.draw_background(painter)
        self.draw_items(painter)
        #self.draw_attention_points(painter)
        self.draw_organisms(painter)

    def draw_background(self, painter):
        painter.fillRect(self.rect(), QColor(*self.BLACK))  # Use BLACK from self

    def draw_items(self, painter):
        # Collect all nearest item IDs from organisms
        nearest_item_ids = set()
        for org_id, org_state in self.sim_state.current_state.get_objects_in_snapshot(ObjectType.ORGANISM):
            if 'nearest_item_ID' in org_state and org_state['nearest_item_ID'] is not None:
                nearest_item_ids.add(org_state['nearest_item_ID'])

        for item_id, item_state in self.sim_state.current_state.get_objects_in_snapshot(ObjectType.ITEM):
            if item_state.get('marked_for_deletion', False):
                raise Exception(f"Item {item_id} is marked for deletion but still present in the state snapshot.")

            screen_x, screen_y = self.grid_to_screen(item_state['x'], item_state['y'])

            # If the item is within the visible area
            if 0 <= screen_x < self.width() and 0 <= screen_y < self.height():
                # If this item is a nearest item for any organism, draw an orange halo
                if item_id in nearest_item_ids:
                    halo_size = self.sim_engine.CELL_SIZE * 3  # 3x3 grid including the item
                    halo_x = screen_x - self.sim_engine.CELL_SIZE
                    halo_y = screen_y - self.sim_engine.CELL_SIZE
                    rect = QRectF(halo_x, halo_y, halo_size, halo_size)
                    painter.fillRect(rect, QColor(*self.ORANGE_HALO))

                # Draw the item
                item_color = item_state['color']  # Should be a tuple like (R, G, B, A)
                color = QColor(*item_color)
                rect = QRectF(screen_x, screen_y, self.sim_engine.CELL_SIZE, self.sim_engine.CELL_SIZE)
                painter.fillRect(rect, color)


    def draw_attention_points(self, painter):
        for org_id, org_state in self.sim_state.current_state.get_objects_in_snapshot(ObjectType.ORGANISM):
            attention_x, attention_y = org_state['attention_x'], org_state['attention_y']
            screen_x, screen_y = self.grid_to_screen(attention_x, attention_y)

            # Calculate the size of the attention point
            attention_size = int(self.sim_engine.CELL_SIZE * 1.0)

            # Calculate the offset to center the attention point
            offset = (attention_size - self.sim_engine.CELL_SIZE) // 2

            # Adjust the position to center the attention point
            centered_x = screen_x - offset
            centered_y = screen_y - offset

            # Draw the attention point
            rect = QRectF(centered_x, centered_y, attention_size, attention_size)
            painter.fillRect(rect, QColor(*self.RED))  # Use RED from self

    def draw_organisms(self, painter):
        for org_id, org_state in self.sim_state.current_state.get_objects_in_snapshot(ObjectType.ORGANISM):
            screen_x, screen_y = self.grid_to_screen(org_state['x'], org_state['y'])
            rect = QRectF(screen_x, screen_y, self.sim_engine.CELL_SIZE, self.sim_engine.CELL_SIZE)
            painter.fillRect(rect, QColor(*self.NEON_GREEN))