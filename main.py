from ui import UI
from organism import Organism
from simulation_engine import SimulationEngine
import time
from simulation_state import SimulationState


def main():
    ui = UI()
    simulation_engine = SimulationEngine(ui)
    
    initial_x, initial_y = simulation_engine.viewport_cell_center_x, simulation_engine.viewport_cell_center_y
    test_organism = simulation_engine.create_organism(Organism, (initial_x, initial_y), simulation_engine.current_state)
    simulation_engine.test_organism = test_organism

    sim_state = SimulationState(simulation_engine, ui)
    
    run_simulation(sim_state)
    
    print_final_summary(sim_state)

def run_simulation(sim_state):
    running = True
    last_print_time = time.time()
    update_times, draw_times, total_frame_times = [], [], []
    frame_count = 0
    last_fps_calc_time = time.time()
    frame_time = 1.0 / sim_state.ui.FPS
    
    while running:
        frame_start = time.time()
        
        update_start_time = time.time()
        sim_state.update()
        update_time = time.time() - update_start_time
        update_times.append(update_time)

        draw_start_time = time.time()
        sim_state.ui.update(sim_state)
        draw_time = time.time() - draw_start_time
        draw_times.append(draw_time)

        total_frame_time = time.time() - frame_start
        total_frame_times.append(total_frame_time)

        frame_count += 1
        current_time = time.time()

        # Calculate FPS every second
        if current_time - last_fps_calc_time >= 1.0:
            sim_state.framerate = frame_count / (current_time - last_fps_calc_time)
            frame_count = 0
            last_fps_calc_time = current_time

        if current_time - last_print_time >= 10:
            print_simulation_summary(sim_state, update_times, draw_times, total_frame_times)
            last_print_time = current_time
            update_times, draw_times, total_frame_times = [], [], []
        
        # Control frame rate
        frame_end = time.time()
        frame_duration = frame_end - frame_start
        if frame_duration < frame_time:
            time.sleep(frame_time - frame_duration)
        
        running = not sim_state.ui.should_exit

def print_simulation_summary(sim_state, update_times, draw_times, total_frame_times):
    print_simulation_stats(sim_state)
    
    avg_update_time = sum(update_times) / max(len(update_times), 1)
    avg_draw_time = sum(draw_times) / max(len(draw_times), 1)
    avg_total_frame_time = sum(total_frame_times) / max(len(total_frame_times), 1)
    
    max_update_time = max(update_times) if update_times else 0
    max_draw_time = max(draw_times) if draw_times else 0
    max_total_frame_time = max(total_frame_times) if total_frame_times else 0
    
    sim_state.avg_update_times.append(avg_update_time)
    sim_state.avg_draw_times.append(avg_draw_time)
    sim_state.avg_total_frame_times.append(avg_total_frame_time)
    
    print(f"Avg Update time: {avg_update_time*1000:.2f}ms (Max: {max_update_time*1000:.2f}ms)")
    print(f"Avg Draw time: {avg_draw_time*1000:.2f}ms (Max: {max_draw_time*1000:.2f}ms)")
    print(f"Avg Total frame time: {avg_total_frame_time*1000:.2f}ms (Max: {max_total_frame_time*1000:.2f}ms)")
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

if __name__ == "__main__":
    main()