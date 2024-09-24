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
    frame_times = []
    frame_count = 0
    last_fps_calc_time = time.time()

    @Slot()
    def simulation_step():
        nonlocal last_print_time, frame_count, last_fps_calc_time
        nonlocal frame_times

        frame_start = time.time()
        
        sim_state.update()
        sim_state.ui.update(sim_state)

        frame_time = time.time() - frame_start
        frame_times.append(frame_time)

        frame_count += 1
        current_time = time.time()

        # Calculate FPS every second
        if current_time - last_fps_calc_time >= 1.0:
            sim_state.framerate = frame_count / (current_time - last_fps_calc_time)
            frame_count = 0
            last_fps_calc_time = current_time

        # Print summary every 10 seconds
        if current_time - last_print_time >= 10:
            print_simulation_summary(sim_state, frame_times)
            last_print_time = current_time
            frame_times = []
        
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

def print_simulation_summary(sim_state, frame_times):
    print_simulation_stats(sim_state)
    
    avg_frame_time = sum(frame_times) / max(len(frame_times), 1)
    max_frame_time = max(frame_times) if frame_times else 0
    
    sim_state.avg_total_frame_times.append(avg_frame_time)
    
    print(f"Avg frame time: {avg_frame_time*1000:.2f}ms (Max: {max_frame_time*1000:.2f}ms)")
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
    
    if sim_state.avg_total_frame_times:
        overall_avg_total_frame_time = sum(sim_state.avg_total_frame_times) / len(sim_state.avg_total_frame_times)
        print(f"Overall average frame time: {overall_avg_total_frame_time*1000:.2f}ms")
    
    print("\n--- Training Statistics ---")
    for stat in sim_state.generate_training_stats():
        print(stat)

if __name__ == "__main__":
    main()