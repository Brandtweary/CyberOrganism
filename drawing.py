import dearpygui.dearpygui as dpg
from state_snapshot import ObjectType


def draw_simulation(layer_id, sim_state):
    draw_background(layer_id, sim_state)
    draw_items(layer_id, sim_state)
    draw_attention_points(layer_id, sim_state)
    draw_organisms(layer_id, sim_state)

def draw_background(layer_id, sim_state):
    dpg.draw_rectangle(
        (0, 0),
        (sim_state.ui.WIDTH, sim_state.ui.HEIGHT),
        parent=layer_id,
        fill=(0, 0, 0, 255),
        thickness=0,
        color=(0, 0, 0, 0)
    )

def draw_items(layer_id, sim_state):
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
            dpg.draw_rectangle(
                (halo_x, halo_y),
                (halo_x + halo_size, halo_y + halo_size),
                parent=layer_id,
                fill=(255, 165, 0, 100), 
                thickness=0,
                color=(0, 0, 0, 0)  # you have to explicitly set the border to transparent
            )
        
        # Draw the item
        dpg.draw_rectangle(
            (screen_x, screen_y),
            (screen_x + sim_state.sim_engine.CELL_SIZE, screen_y + sim_state.sim_engine.CELL_SIZE),
            parent=layer_id,
            fill=item_state['color'],
            thickness=0,
            color=(0, 0, 0, 0)
        )

def draw_attention_points(layer_id, sim_state):
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
        dpg.draw_rectangle(
            (centered_x, centered_y),
            (centered_x + attention_size, centered_y + attention_size),
            parent=layer_id,
            fill=sim_state.sim_engine.RED,
            thickness=0,
            color=(0, 0, 0, 0)
        )

def draw_organisms(layer_id, sim_state):
    for org_id, org_state in sim_state.current_state.get_objects_in_snapshot(ObjectType.ORGANISM):
        screen_x, screen_y = sim_state.sim_engine.grid_to_screen(org_state['x'], org_state['y'])
        dpg.draw_rectangle(
            (screen_x, screen_y),
            (screen_x + sim_state.sim_engine.CELL_SIZE, screen_y + sim_state.sim_engine.CELL_SIZE),
            parent=layer_id,
            fill=sim_state.sim_engine.NEON_GREEN,
            thickness=0,
            color=(0, 0, 0, 0)
        )
