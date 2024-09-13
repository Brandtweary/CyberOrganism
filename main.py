from ui import UI
import pygame
from organism import Organism
from simulation_engine import SimulationEngine
import time
from drawing import draw_simulation
from simulation_state import SimulationState


def main():
    ui = UI()
    display, clock = ui.create_window()
    
    simulation_engine = SimulationEngine()
    
    initial_x, initial_y = simulation_engine.GRID_SIZE // 2, simulation_engine.GRID_SIZE // 2
    test_organism = simulation_engine.create_organism(Organism, (initial_x, initial_y), simulation_engine.current_state)
    simulation_engine.test_organism = test_organism

    sim_state = SimulationState(simulation_engine, clock, ui)
    
    run_simulation(ui, display, clock, sim_state)
    
    pygame.quit()

def run_simulation(ui, display, clock, sim_state):
    running = True
    last_print_time = time.time()
    update_times, draw_times, total_frame_times = [], [], []
    frame_count = 0
    
    while running:
        running = ui.handle_events()
        if not running:
            break
        
        if sim_state.total_time >= 0.5:
            sim_state.sim_engine.handle_camera_panning()
        
        update_start_time = time.time()
        if frame_count % 2 == 0:
            sim_state.update()
            update_times.append(time.time() - update_start_time)

        draw_start_time = time.time()
        draw_simulation(display, sim_state)
        pygame.display.flip()
        draw_times.append(time.time() - draw_start_time)

        if frame_count % 2 == 0:
            total_frame_times.append(time.time() - update_start_time)

        current_time = time.time()
        if current_time - last_print_time >= 10:
            print_simulation_summary(sim_state, update_times, draw_times, total_frame_times)
            last_print_time = current_time
            update_times, draw_times, total_frame_times = [], [], []

        clock.tick(60)
        frame_count += 1
    
    print_final_summary(sim_state)

def print_simulation_summary(sim_state, update_times, draw_times, total_frame_times):
    print_simulation_stats(sim_state)
    
    avg_update_time = sum(update_times) / max(len(update_times), 1)
    avg_draw_time = sum(draw_times) / max(len(draw_times), 1)
    avg_total_frame_time = sum(total_frame_times) / max(len(total_frame_times), 1)
    
    sim_state.avg_update_times.append(avg_update_time)
    sim_state.avg_draw_times.append(avg_draw_time)
    sim_state.avg_total_frame_times.append(avg_total_frame_time)
    
    print(f"Avg Update time: {avg_update_time*1000:.2f}ms")
    print(f"Avg Draw time: {avg_draw_time*1000:.2f}ms")
    print(f"Avg Total frame time: {avg_total_frame_time*1000:.2f}ms")
    current_overall_avg = sum(sim_state.avg_update_times) / len(sim_state.avg_update_times)
    print(f"Current overall avg update time: {current_overall_avg*1000:.2f}ms")
    print("--------------------")

def print_simulation_stats(sim_state):
    print("\n--- Organism Statistics ---")
    for stat in sim_state.generate_organism_stats():
        print(stat)
    print("\n--- Performance Statistics ---")
    for stat in sim_state.generate_performance_stats():
        print(stat)
    print("\n--- Training Statistics ---")
    for stat in sim_state.generate_training_stats():
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
    for stat in sim_state.generate_training_stats():
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