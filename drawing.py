import pygame
from state_snapshot import ObjectType
import time


def draw_simulation(screen, sim_state):
    screen.fill(sim_state.sim_engine.BLACK)
    draw_items(screen, sim_state)
    draw_attention_points(screen, sim_state)
    draw_organisms(screen, sim_state)
    draw_gui(screen, sim_state.ui, sim_state)

def draw_items(screen, sim_state):
    # Collect all nearest item IDs from organisms
    nearest_item_ids = set()
    for org_id, org_state in sim_state.current_state.get_objects_in_snapshot(ObjectType.ORGANISM):
        if 'nearest_item_ID' in org_state and org_state['nearest_item_ID'] is not None:
            nearest_item_ids.add(org_state['nearest_item_ID'])

    for item_id, item_state in sim_state.current_state.get_objects_in_snapshot(ObjectType.ITEM):
        if item_state.get('marked_for_deletion', False):
            raise Exception(f"Item {item_id} is marked for deletion but still present in the state snapshot.")
        
        screen_x, screen_y = sim_state.sim_engine.grid_to_screen(item_state['x'], item_state['y'])
        
        # If this item is a nearest item for any organism, draw an orange halo
        if item_id in nearest_item_ids:
            halo_size = sim_state.sim_engine.CELL_SIZE * 3  # 3x3 grid including the item
            halo_x = screen_x - sim_state.sim_engine.CELL_SIZE
            halo_y = screen_y - sim_state.sim_engine.CELL_SIZE
            pygame.draw.rect(screen, (255, 165, 0),  # Orange color
                             (halo_x, halo_y, halo_size, halo_size))
        
        # Draw the item
        pygame.draw.rect(screen, item_state['color'], (screen_x, screen_y, sim_state.sim_engine.CELL_SIZE, sim_state.sim_engine.CELL_SIZE))


def draw_attention_points(screen, sim_state):
    for org_id, org_state in sim_state.current_state.get_objects_in_snapshot(ObjectType.ORGANISM):
        attention_x, attention_y = org_state['attention_x'], org_state['attention_y']
        screen_x, screen_y = sim_state.sim_engine.grid_to_screen(attention_x, attention_y)
        
        # Calculate the size of the attention point
        attention_size = int(sim_state.sim_engine.CELL_SIZE * 1.0)
        
        # Calculate the offset to center the attention point
        offset = (attention_size - sim_state.sim_engine.CELL_SIZE) // 2
        
        # Adjust the position to center the attention point
        centered_x = screen_x - offset
        centered_y = screen_y - offset
        
        # Draw the attention point
        pygame.draw.rect(screen, sim_state.sim_engine.RED, 
                            (centered_x, centered_y, attention_size, attention_size))


def draw_organisms(screen, sim_state):
    for org_id, org_state in sim_state.current_state.get_objects_in_snapshot(ObjectType.ORGANISM):
        screen_x, screen_y = sim_state.sim_engine.grid_to_screen(org_state['x'], org_state['y'])
        if 0 <= screen_x < sim_state.sim_engine.SCREEN_WIDTH and 0 <= screen_y < sim_state.sim_engine.SCREEN_HEIGHT:
            rect = pygame.Rect(screen_x, screen_y, sim_state.sim_engine.CELL_SIZE, sim_state.sim_engine.CELL_SIZE)
            pygame.draw.rect(screen, sim_state.sim_engine.NEON_GREEN, rect)

def draw_gui(screen, ui):
    ui_elements = ui.get_ui_elements()
    
    # Draw background elements
    for element_name, element in ui_elements.items():
        background_color = element.get('background_color', (0, 0, 0, 0))
        if background_color[3] > 0:
            pygame.draw.rect(screen, background_color, 
                             (element['x'], element['y'], element['width'], element['height']))

    # Draw text elements
    text_elements = []
    for element_name, element in ui_elements.items():
        if element['container_type'] == 'text':
            text_elements.append((
                element['text'],
                element['font'],
                element['font_color'],
                (element['x'] + element['inner_margin'], element['y'] + element['inner_margin'])
            ))

    # Batch render text
    for text, font, color, position in text_elements:
        text_surface = font.render(text, True, color)
        screen.blit(text_surface, position)
