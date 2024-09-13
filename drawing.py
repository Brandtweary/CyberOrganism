import pygame
from state_snapshot import ObjectType
import time


class CachedElement:
    def __init__(self, surface):
        self.surface = surface

cached_surfaces = {}

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

def draw_gui(screen, ui, sim_state):
    ui_elements = ui.get_ui_elements()
    
    if sim_state.total_time >= 1.0:  # Only profile after 1 second has passed
        start_time = time.perf_counter()
        draw_element(screen, "root", ui_elements, 0, 0, profile=True)
        end_time = time.perf_counter()
        total_render_time_ms = (end_time - start_time) * 1000
        print(f"Total GUI render time: {total_render_time_ms:.2f} ms")
        
        # Breakpoint spot for pausing execution
        breakpoint_here = True  # You can set a breakpoint on this line
    else:
        draw_element(screen, "root", ui_elements, 0, 0, profile=False)

def draw_element(screen, element_name, all_elements, x, y, profile=False):
    element = all_elements[element_name]
    content_x = x + element["inner_margin"]
    content_y = y + element["inner_margin"]

    if profile:
        start_time = time.perf_counter()

    # Check if we have a cached surface for this element
    if element_name in cached_surfaces:
        # Blit the cached surface (background and static content)
        screen.blit(cached_surfaces[element_name].surface, (x, y))
    else:
        # Draw background if color has non-zero alpha
        background_color = element.get('background_color', (0, 0, 0, 0))
        if background_color[3] > 0:
            pygame.draw.rect(screen, background_color, (x, y, element["width"], element["height"]))

        # Draw text for text elements
        if element["container_type"] == "text":
            font = element.get('font', pygame.font.Font(None, 24))
            font_color = element.get('font_color', (255, 255, 255))
            text_surface = font.render(element["text"], True, font_color)
            screen.blit(text_surface, (content_x, content_y))

        # Cache the element if it's cacheable
        if element.get('cacheable', False):
            cache_surface = pygame.Surface((element["width"], element["height"]), pygame.SRCALPHA)
            cache_surface.blit(screen, (0, 0), (x, y, element["width"], element["height"]))
            cached_surfaces[element_name] = CachedElement(cache_surface)

    # Print rendering time for this element
    if profile:
        end_time = time.perf_counter()
        render_time_ms = (end_time - start_time) * 1000
        print(f"Element {element_name} render time: {render_time_ms:.2f} ms")

    # Process child elements
    if element["container_type"] in ["vbox", "hbox"]:
        child_x = content_x
        child_y = content_y
        
        for child_name in element["child_elements"]:
            child = all_elements[child_name]
            
            # Consider the child's outer margin when positioning
            draw_element(screen, child_name, all_elements, 
                         child_x + child["outer_margin"], 
                         child_y + child["outer_margin"],
                         profile=profile)
            
            if element["container_type"] == "vbox":
                child_y += child["height"] + child["outer_margin"] * 2
            elif element["container_type"] == "hbox":
                child_x += child["width"] + child["outer_margin"] * 2

def clear_cache():
    cached_surfaces.clear()
