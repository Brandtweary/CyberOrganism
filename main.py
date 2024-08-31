import pygame
import numpy as np
import psutil
from organism import Organism
from matrika import Matrika
import time
import math


class SimulationState:
    def __init__(self, matrika):
        self.matrika = matrika
        self.current_state = matrika.current_state
        self.test_organism = matrika.test_organism
        self.input_parameters = {param: getattr(self.test_organism, param) for param in self.test_organism.input_parameters}
        self.display_parameters = {param: getattr(self.test_organism, param) for param in self.test_organism.display_parameters}
        self.cpu_usage = psutil.cpu_percent()
        self.memory_usage = psutil.virtual_memory().percent
        self.available_memory = psutil.virtual_memory().available / (1024 * 1024)
        self.update_fps = 0
        
        # Summary statistics
        self.time_low_loss = 0
        self.start_time = time.time()
        self.total_time = 0
        
        self.frame_times = []
        
        # FPS calculation
        self.frame_count = 0
        self.last_fps_calc_time = self.start_time

        # Add training statistics
        self.training_stats = self.test_organism.training_stats.get_stats()
        self.training_record_stats = self.test_organism.training_stats.record_stats

        # Add new attributes for storing average times
        self.avg_update_times = []
        self.avg_draw_times = []
        self.avg_total_frame_times = []

        # Add last update time for interpolation
        self.last_update_time = time.time()

    def update(self):
        cycle_start_time = time.time()
        
        self.matrika.update_simulation()
        self.current_state = self.matrika.current_state
        self.input_parameters = {param: getattr(self.test_organism, param) for param in self.test_organism.input_parameters}
        self.display_parameters = {param: getattr(self.test_organism, param) for param in self.test_organism.display_parameters}
        
        # Update summary statistics
        current_loss = self.display_parameters['loss_avg']
        if current_loss < 0.5:
            self.time_low_loss += self.matrika.UPDATE_INTERVAL
        self.total_time = time.time() - self.start_time
        
        # Update performance metrics
        self.cpu_usage = psutil.cpu_percent()
        self.memory_usage = psutil.virtual_memory().percent
        self.available_memory = psutil.virtual_memory().available / (1024 * 1024)
        
        # Calculate FPS
        cycle_end_time = time.time()
        self.frame_times.append(cycle_end_time - cycle_start_time)
        if len(self.frame_times) > 100:
            self.frame_times.pop(0)
        
        # Update FPS calculation
        self.frame_count += 1
        self.calculate_fps()

        # Update training statistics
        self.training_stats = self.test_organism.training_stats.get_stats()
        self.training_record_stats = self.test_organism.training_stats.record_stats

        # Update last update time for interpolation
        self.last_update_time = time.time()

    def calculate_fps(self):
        current_time = time.time()
        elapsed_time = current_time - self.last_fps_calc_time
        
        if elapsed_time >= 1.0:
            self.update_fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.last_fps_calc_time = current_time

def main():
    pygame.init()
    
    matrika = Matrika()  # Create Matrika instance
    
    screen = pygame.display.set_mode((matrika.SCREEN_WIDTH, matrika.SCREEN_HEIGHT), pygame.FULLSCREEN)
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)

    # Create the test organism at the center of the grid
    initial_x, initial_y = matrika.GRID_SIZE // 2, matrika.GRID_SIZE // 2
    test_organism = matrika.create_organism(Organism, initial_x, initial_y)
    matrika.test_organism = test_organism

    sim_state = SimulationState(matrika)
    
    run_simulation(screen, clock, font, sim_state)
    
    pygame.quit()

def run_simulation(screen, clock, font, sim_state):
    running = True
    last_print_time = time.time()
    update_times = []
    draw_times = []
    total_frame_times = []
    frame_count = 0
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        if not running:
            break

        # Handle camera panning (60 FPS) only after 0.5 seconds
        if sim_state.total_time >= 0.5:
            sim_state.matrika.handle_camera_panning()

        # Update simulation (30 FPS)
        update_start_time = time.time()
        if frame_count % 2 == 0:
            sim_state.update()
            update_end_time = time.time()
            update_time = update_end_time - update_start_time
            update_times.append(update_time)

        # Draw (60 FPS)
        draw_start_time = time.time()
        draw_simulation(screen, sim_state, font, clock)
        pygame.display.flip()
        draw_end_time = time.time()

        # Calculate timings
        draw_time = draw_end_time - draw_start_time
        draw_times.append(draw_time)

        # Only record total frame time for update frames
        if frame_count % 2 == 0:
            total_frame_time = time.time() - update_start_time
            total_frame_times.append(total_frame_time)

        current_time = time.time()
        if current_time - last_print_time >= 10:
            print_biomarkers(sim_state, clock.get_fps())
            
            avg_update_time = sum(update_times) / len(update_times)
            avg_draw_time = sum(draw_times) / len(draw_times)
            avg_total_frame_time = sum(total_frame_times) / len(total_frame_times)
            
            # Store the average times in sim_state
            sim_state.avg_update_times.append(avg_update_time)
            sim_state.avg_draw_times.append(avg_draw_time)
            sim_state.avg_total_frame_times.append(avg_total_frame_time)
            
            print(f"Avg Update time: {avg_update_time*1000:.2f}ms")
            print(f"Avg Draw time: {avg_draw_time*1000:.2f}ms")
            print(f"Avg Total frame time: {avg_total_frame_time*1000:.2f}ms")
            print("--------------------")
            
            last_print_time = current_time
            update_times.clear()
            draw_times.clear()
            total_frame_times.clear()

        clock.tick(60)  # Set to 60 FPS
        frame_count += 1
    
    print_final_summary(sim_state)

def draw_simulation(screen, sim_state, font, clock):
    screen.fill(sim_state.matrika.BLACK)
    draw_items(screen, sim_state)
    draw_attention_points(screen, sim_state)
    draw_organisms(screen, sim_state)
    display_biomarkers(screen, font, clock, sim_state)

def draw_items(screen, sim_state):
    for item_id, item in sim_state.current_state['items'].items():
        screen_x, screen_y = sim_state.matrika.grid_to_screen(item['x'], item['y'])
        pygame.draw.rect(screen, item['color'], (screen_x, screen_y, sim_state.matrika.CELL_SIZE, sim_state.matrika.CELL_SIZE))


def draw_attention_points(screen, sim_state):
    for organism in sim_state.current_state['organisms'].values():
        attention_x, attention_y = organism['attention_point']
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
    for organism_id, organism in sim_state.current_state['organisms'].items():
        screen_x, screen_y = sim_state.matrika.grid_to_screen(organism['x'], organism['y'])
        if 0 <= screen_x < sim_state.matrika.SCREEN_WIDTH and 0 <= screen_y < sim_state.matrika.SCREEN_HEIGHT:
            rect = pygame.Rect(screen_x, screen_y, sim_state.matrika.CELL_SIZE, sim_state.matrika.CELL_SIZE)
            pygame.draw.rect(screen, sim_state.matrika.NEON_GREEN, rect)


def format_biomarker_name(name):
    if name is None:
        return "None"
    try:
        return ' '.join(word.capitalize() for word in str(name).split('_'))
    except AttributeError:
        return str(name)

def display_biomarkers(screen, font, clock, sim_state):
    """
    Display input parameters, display parameters, and performance data on the screen.
    """
    # Prepare parameter text
    parameter_text = []
    for param, value in sim_state.input_parameters.items():
        formatted_name = format_biomarker_name(param)
        parameter_text.append(f"{formatted_name}: {value}")
    
    for param, value in sim_state.display_parameters.items():
        formatted_name = format_biomarker_name(param)
        parameter_text.append(f"{formatted_name}: {value}")

    # Prepare performance text
    performance_text = [
        f"CPU Usage: {sim_state.cpu_usage}%",
        f"Memory Usage: {sim_state.memory_usage}%",
        f"Available Memory: {sim_state.available_memory:.1f} MB",
        f"Display FPS: {clock.get_fps():.1f}",
        f"Update FPS: {sim_state.update_fps:.1f}",
        f"Simulation Time: {int(sim_state.total_time)} seconds",
    ]

    # Combine parameter and performance text
    all_text = parameter_text + [""] + performance_text

    # Render and display text
    text_y = 10
    for text in all_text:
        text_surface = font.render(text, True, sim_state.matrika.NEON_GREEN, sim_state.matrika.BLACK)
        text_rect = text_surface.get_rect()
        text_rect.topright = (sim_state.matrika.SCREEN_WIDTH - 10, text_y)
        screen.blit(text_surface, text_rect)
        text_y += 30

def print_biomarkers(sim_state, display_fps):
    print("\n--- Parameters ---")
    
    for name, value in sim_state.input_parameters.items():
        formatted_name = format_biomarker_name(name)
        print(f"{formatted_name}: {value}")
    
    for name, value in sim_state.display_parameters.items():
        formatted_name = format_biomarker_name(name)
        print(f"{formatted_name}: {value}")
    
    print("---------------------------------------")
    
    print("--- Performance Data ---")
    print(f"CPU Usage: {sim_state.cpu_usage}%")
    print(f"Memory Usage: {sim_state.memory_usage}%")
    print(f"Available Memory: {sim_state.available_memory:.1f} MB")
    print(f"Update FPS: {sim_state.update_fps:.1f}")
    print(f"Display FPS: {display_fps:.1f}")
    print(f"Simulation Time: {int(sim_state.total_time)} seconds")
    print("---------------------------------------")

    print_training_statistics(sim_state)

def print_final_summary(sim_state):
    print("\n=== Simulation Summary ===")
    print(f"Time spent with total loss < 0.5: {sim_state.time_low_loss:.2f} seconds")
    print(f"Percentage of time with low loss: {(sim_state.time_low_loss / sim_state.total_time) * 100:.2f}%")
    print(f"Total simulation time: {sim_state.total_time:.2f} seconds")
    
    # Add overall average time calculations
    if sim_state.avg_update_times:
        overall_avg_update_time = sum(sim_state.avg_update_times) / len(sim_state.avg_update_times)
        overall_avg_draw_time = sum(sim_state.avg_draw_times) / len(sim_state.avg_draw_times)
        overall_avg_total_frame_time = sum(sim_state.avg_total_frame_times) / len(sim_state.avg_total_frame_times)
        
        print(f"\nOverall Average Times:")
        print(f"  Update time: {overall_avg_update_time*1000:.2f}ms")
        print(f"  Draw time: {overall_avg_draw_time*1000:.2f}ms")
        print(f"  Total frame time: {overall_avg_total_frame_time*1000:.2f}ms")
    
    print_training_statistics(sim_state)

def print_training_statistics(sim_state):
    if any(stat_info['record'] for stat_info in sim_state.training_record_stats.values()):
        print("\nTraining Statistics:")
        for stat_type, stat_info in sim_state.training_record_stats.items():
            if stat_info['record']:
                print(f"  {stat_type}:")
                for stat_name in stat_info['stat_names']:
                    if stat_name in sim_state.training_stats:
                        values = sim_state.training_stats[stat_name]
                        if values:
                            value = values[-1]
                            if isinstance(value, (int, float)):
                                print(f"    {stat_name}: {value:.4f}")
                            else:
                                print(f"    {stat_name}: {value}")

def handle_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return False
    return True

if __name__ == "__main__":
    main()