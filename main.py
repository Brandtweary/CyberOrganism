from ui import UI
from organism import Organism
from simulation_engine import SimulationEngine
import time
from simulation_state import SimulationState
from PySide6.QtCore import QTimer, QEventLoop, QCoreApplication, QThread, Signal, Slot


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
    event_loop = QEventLoop()
    timer_thread = TimerThread(1 / sim_state.ui.FPS)
    
    last_print_time = time.time()
    update_times, draw_times, total_frame_times, timing_deviations, event_backlogs = [], [], [], [], []
    frame_count = 0
    last_fps_calc_time = time.time()
    last_timer_call = time.time()

    @Slot()
    def simulation_step():
        nonlocal last_print_time, frame_count, last_fps_calc_time, last_timer_call
        nonlocal update_times, draw_times, total_frame_times, timing_deviations, event_backlogs

        current_time = time.time()
        actual_interval = current_time - last_timer_call
        expected_interval = 1 / sim_state.ui.FPS
        timing_deviations.append((actual_interval - expected_interval) * 1000)  # Convert to milliseconds
        last_timer_call = current_time

        frame_start = time.time()

        # Process pending events and measure the time
        start_process = time.time()
        QCoreApplication.processEvents()
        event_processing_time = time.time() - start_process
        event_backlogs.append(event_processing_time * 1000)  # Convert to milliseconds
        
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

        # Print summary every 10 seconds
        if current_time - last_print_time >= 10:
            print_simulation_summary(sim_state, update_times, draw_times, total_frame_times, timing_deviations, event_backlogs)
            last_print_time = current_time
            update_times, draw_times, total_frame_times, timing_deviations, event_backlogs = [], [], [], [], []
        
        if sim_state.ui.should_exit:
            timer_thread.stop()
            event_loop.quit()

    timer_thread.timeout.connect(simulation_step)
    timer_thread.start()

    event_loop.exec()
    timer_thread.wait()  # Wait for the thread to finish

class TimerThread(QThread):
    timeout = Signal()

    def __init__(self, interval_seconds):
        super().__init__()
        self.interval = interval_seconds
        self.running = True

    def run(self):
        while self.running:
            time.sleep(self.interval)
            self.timeout.emit()

    def stop(self):
        self.running = False

def print_simulation_summary(sim_state, update_times, draw_times, total_frame_times, timing_deviations, event_backlogs):
    print_simulation_stats(sim_state)
    
    avg_update_time = sum(update_times) / max(len(update_times), 1)
    avg_draw_time = sum(draw_times) / max(len(draw_times), 1)
    avg_total_frame_time = sum(total_frame_times) / max(len(total_frame_times), 1)
    avg_timing_deviation = sum(timing_deviations) / max(len(timing_deviations), 1)
    avg_event_backlog = sum(event_backlogs) / max(len(event_backlogs), 1)
    
    max_update_time = max(update_times) if update_times else 0
    max_draw_time = max(draw_times) if draw_times else 0
    max_total_frame_time = max(total_frame_times) if total_frame_times else 0
    max_timing_deviation = max(timing_deviations) if timing_deviations else 0
    max_event_backlog = max(event_backlogs) if event_backlogs else 0
    
    sim_state.avg_update_times.append(avg_update_time)
    sim_state.avg_draw_times.append(avg_draw_time)
    sim_state.avg_total_frame_times.append(avg_total_frame_time)
    
    print(f"Avg Update time: {avg_update_time*1000:.2f}ms (Max: {max_update_time*1000:.2f}ms)")
    print(f"Avg Draw time: {avg_draw_time*1000:.2f}ms (Max: {max_draw_time*1000:.2f}ms)")
    print(f"Avg Total frame time: {avg_total_frame_time*1000:.2f}ms (Max: {max_total_frame_time*1000:.2f}ms)")
    print(f"Avg Event backlog: {avg_event_backlog:.2f}ms (Max: {max_event_backlog:.2f}ms)")
    print(f"Avg Timing deviation: {avg_timing_deviation:.2f}ms (Max: {max_timing_deviation:.2f}ms)")
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