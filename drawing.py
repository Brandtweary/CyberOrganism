import pygame
from state_snapshot import ObjectType


def draw_simulation(screen, sim_state, font, clock):
    screen.fill(sim_state.matrika.BLACK)
    draw_items(screen, sim_state)
    draw_attention_points(screen, sim_state)
    draw_organisms(screen, sim_state)
    display_simulation_stats(screen, font, clock, sim_state)

def draw_items(screen, sim_state):
    # Collect all nearest item IDs from organisms
    nearest_item_ids = set()
    for org_id, org_state in sim_state.current_state.get_objects_in_snapshot(ObjectType.ORGANISM):
        if 'nearest_item_ID' in org_state and org_state['nearest_item_ID'] is not None:
            nearest_item_ids.add(org_state['nearest_item_ID'])

    for item_id, item_state in sim_state.current_state.get_objects_in_snapshot(ObjectType.ITEM):
        if item_state.get('marked_for_deletion', False):
            raise Exception(f"Item {item_id} is marked for deletion but still present in the state snapshot.")
        
        screen_x, screen_y = sim_state.matrika.grid_to_screen(item_state['x'], item_state['y'])
        
        # If this item is a nearest item for any organism, draw an orange halo
        if item_id in nearest_item_ids:
            halo_size = sim_state.matrika.CELL_SIZE * 3  # 3x3 grid including the item
            halo_x = screen_x - sim_state.matrika.CELL_SIZE
            halo_y = screen_y - sim_state.matrika.CELL_SIZE
            pygame.draw.rect(screen, (255, 165, 0),  # Orange color
                             (halo_x, halo_y, halo_size, halo_size))
        
        # Draw the item
        pygame.draw.rect(screen, item_state['color'], (screen_x, screen_y, sim_state.matrika.CELL_SIZE, sim_state.matrika.CELL_SIZE))


def draw_attention_points(screen, sim_state):
    for org_id, org_state in sim_state.current_state.get_objects_in_snapshot(ObjectType.ORGANISM):
        if 'attention_point' in org_state:
            attention_x, attention_y = org_state['attention_point']
            screen_x, screen_y = sim_state.matrika.grid_to_screen(attention_x, attention_y)
            
            # Calculate the size of the attention point
            attention_size = int(sim_state.matrika.CELL_SIZE * 1.5)
            
            # Calculate the offset to center the attention point
            offset = (attention_size - sim_state.matrika.CELL_SIZE) // 2
            
            # Adjust the position to center the attention point
            centered_x = screen_x - offset
            centered_y = screen_y - offset
            
            # Draw the attention point
            pygame.draw.rect(screen, sim_state.matrika.RED, 
                             (centered_x, centered_y, attention_size, attention_size))


def draw_organisms(screen, sim_state):
    for org_id, org_state in sim_state.current_state.get_objects_in_snapshot(ObjectType.ORGANISM):
        screen_x, screen_y = sim_state.matrika.grid_to_screen(org_state['x'], org_state['y'])
        if 0 <= screen_x < sim_state.matrika.SCREEN_WIDTH and 0 <= screen_y < sim_state.matrika.SCREEN_HEIGHT:
            rect = pygame.Rect(screen_x, screen_y, sim_state.matrika.CELL_SIZE, sim_state.matrika.CELL_SIZE)
            pygame.draw.rect(screen, sim_state.matrika.NEON_GREEN, rect)

def display_simulation_stats(screen, font, clock, sim_state):
    simulation_stats = sim_state.generate_simulation_statistics()
    simulation_stats.append(f"Display FPS: {clock.get_fps():.1f}")
    
    text_y = 10
    for text in simulation_stats:
        text_surface = font.render(text, True, sim_state.matrika.NEON_GREEN, sim_state.matrika.BLACK)
        text_rect = text_surface.get_rect()
        text_rect.topright = (sim_state.matrika.SCREEN_WIDTH - 10, text_y)
        screen.blit(text_surface, text_rect)
        text_y += 30