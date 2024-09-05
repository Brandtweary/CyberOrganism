import pygame
from organism import Organism
from simulation_engine import SimulationEngine
import time
from drawing import draw_simulation
from simulation_state import SimulationState


def main():
    pygame.init()
    
    simulation_engine = SimulationEngine()  # Create simulation_engine instance
    
    screen = pygame.display.set_mode((simulation_engine.SCREEN_WIDTH, simulation_engine.SCREEN_HEIGHT), pygame.FULLSCREEN)
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)

    # Create the test organism at the center of the grid
    initial_x, initial_y = simulation_engine.GRID_SIZE // 2, simulation_engine.GRID_SIZE // 2
    test_organism = simulation_engine.create_organism(Organism, (initial_x, initial_y), simulation_engine.current_state)
    simulation_engine.test_organism = test_organism

    sim_state = SimulationState(simulation_engine)
    
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
            sim_state.sim_engine.handle_camera_panning()

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
            print_simulation_stats(sim_state, clock.get_fps())
            
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
            current_overall_avg = sum(sim_state.avg_update_times) / len(sim_state.avg_update_times)
            print(f"Current overall avg update time: {current_overall_avg*1000:.2f}ms")  # For debugging
            print("--------------------")
            
            last_print_time = current_time
            update_times.clear()
            draw_times.clear()
            total_frame_times.clear()

        clock.tick(60)  # Set to 60 FPS
        frame_count += 1
    
    print_final_summary(sim_state)

def print_simulation_stats(sim_state, display_fps):
    print("\n--- Simulation Statistics ---")
    for stat in sim_state.generate_simulation_statistics():
        print(stat)
    print(f"Display FPS: {display_fps:.1f}")
    print("---------------------------------------")

    print("\n--- Training Statistics ---")
    for stat in sim_state.generate_training_statistics():
        print(stat)
    print("---------------------------------------")

def print_final_summary(sim_state):
    print("\n=== Simulation Summary ===")
    print(f"Time spent with total loss < 0.5: {sim_state.time_low_loss:.2f} seconds")
    print(f"Percentage of time with low loss: {(sim_state.time_low_loss / sim_state.total_time) * 100:.2f}%")
    print(f"Total simulation time: {sim_state.total_time:.2f} seconds")
    
    if sim_state.avg_update_times:
        overall_avg_update_time = sum(sim_state.avg_update_times) / len(sim_state.avg_update_times)
        overall_avg_draw_time = sum(sim_state.avg_draw_times) / len(sim_state.avg_draw_times)
        overall_avg_total_frame_time = sum(sim_state.avg_total_frame_times) / len(sim_state.avg_total_frame_times)
        
        print(f"\nOverall Average Times:")
        print(f"  Update time: {overall_avg_update_time*1000:.2f}ms")
        print(f"  Draw time: {overall_avg_draw_time*1000:.2f}ms")
        print(f"  Total frame time: {overall_avg_total_frame_time*1000:.2f}ms")
    
    print("\n--- Training Statistics ---")
    for stat in sim_state.generate_training_statistics():
        print(stat)

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