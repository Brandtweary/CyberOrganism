from ui import UI
from simulation_engine import SimulationEngine
import time
from simulation_state import SimulationState
from PySide6.QtCore import QEventLoop, QThread, Signal, Slot
from shared_resources import debug
from summary_logger import summary_logger
from custom_profiler import profiler


def main():
    from process_pool import ProcessPool
    process_pool = ProcessPool(num_processes=4)
    process_pool.start()
    ui = UI()
    simulation_engine = SimulationEngine(ui, process_pool)
    sim_state = SimulationState(simulation_engine, ui)
    run_simulation(sim_state) # anything placed after this will execute after the simulation has finished

def run_simulation(sim_state):
    event_loop = QEventLoop()
    timer_thread = TimerThread(1 / sim_state.ui.FPS)
    
    last_print_time = time.time()
    frame_times = []
    frame_count = 0
    last_fps_calc_time = time.time()
    start_time = time.time()

    @profiler.profile("simulation_step")
    @Slot()
    def simulation_step():
        nonlocal last_print_time, frame_count, last_fps_calc_time, start_time
        nonlocal frame_times, sim_state

        frame_start = time.time()
  
        #with profiler.profile_section("simulation_step", "processEvents"):
            #sim_state.ui.app.processEvents()  # enable to profile Qt event processing, disable to process events during downtime
        with profiler.profile_section("simulation_step", "sim_state_update"):
            sim_state.update()
        with profiler.profile_section("simulation_step", "ui_update"):
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

        # Regular summary every 10 seconds
        if current_time - last_print_time >= 10.0:
            print_simulation_summary(sim_state, frame_times)
            last_print_time = current_time
            frame_times = []
        
        if sim_state.ui.should_exit or sim_state.sim_engine.stop_simulation:
            event_loop.quit()


    timer_thread.timeout.connect(simulation_step)
    timer_thread.start()

    try:
        event_loop.exec()
    except Exception as e:
        print(f"Simulation halted due to exception: {e}")
    finally:
        cleanup_simulation(timer_thread, sim_state)

def cleanup_simulation(timer_thread, sim_state):
    timer_thread.stop()
    sim_state.cleanup()
    print_final_summary(sim_state)
    sim_state.sim_engine.cleanup()

class TimerThread(QThread):
    timeout = Signal()

    def __init__(self, interval_seconds):
        super().__init__()
        self.interval = interval_seconds
        self.running = True

    def run(self):
        last_time = time.perf_counter()
        while self.running:
            current_time = time.perf_counter()
            elapsed = current_time - last_time
            
            # Calculate sleep time, adjusting for longer frames but not shorter ones
            if elapsed > self.interval:
                sleep_time = max(0, self.interval - (elapsed - self.interval))
            else:
                sleep_time = self.interval
            
            time.sleep(sleep_time)
            self.timeout.emit()
            
            last_time = current_time

    def stop(self):
        self.running = False

def print_simulation_summary(sim_state, frame_times):
    print_simulation_stats(sim_state)
    
    avg_frame_time = sum(frame_times) / max(len(frame_times), 1)
    max_frame_time = max(frame_times) if frame_times else 0
    
    sim_state.avg_total_frame_times.append(avg_frame_time)
    if sim_state.total_time > 11:
        sim_state.max_frame_time = max(sim_state.max_frame_time, max_frame_time)
    
    print(f"Avg frame time: {avg_frame_time*1000:.2f}ms (Max: {max_frame_time*1000:.2f}ms)")
    
    # Print custom profiler stats
    print("\nFunction Times:")
    print(profiler.get_performance_stats())
    
    # Print periodic summary for test organism
    print("\n=== Logged Messages ===")
    print(summary_logger.get_periodic_summary(sim_state.test_organism.id))
    
    # Reset profiler stats
    profiler.reset_stats()
    
    print("--------------------")

def format_stats(stats_list):
    return [f"{key}: {value}" for key, value in stats_list]

def print_simulation_stats(sim_state):
    print("\n--- Organism Statistics ---")
    for stat in format_stats(sim_state.generate_organism_stats()):
        print(stat)
    print("\n--- Performance Statistics ---")
    for stat in format_stats(sim_state.generate_performance_stats()):
        print(stat)
    print("\n--- Network Statistics ---")
    for stat in format_stats(sim_state.generate_network_stats()):
        print(stat)
    print("---------------------------------------")

def print_final_summary(sim_state):
    print("\n=== Simulation Summary ===")
    print(f"Time spent with total loss < 0.5: {sim_state.time_low_loss:.2f} seconds")
    print(f"Percentage of time with low loss: {(sim_state.time_low_loss / sim_state.total_time) * 100:.2f}%")
    print(f"Total simulation time: {int(sim_state.total_time)} seconds")
    
    if sim_state.avg_total_frame_times:
        overall_avg_total_frame_time = sum(sim_state.avg_total_frame_times) / len(sim_state.avg_total_frame_times)
        print(f"Overall average frame time: {overall_avg_total_frame_time*1000:.2f}ms (Max: {sim_state.max_frame_time*1000:.2f}ms)")
    
    print("\n--- Network Statistics ---")
    for stat in format_stats(sim_state.generate_network_stats()):
        print(stat)
    
    print("\n=== Logged Messages ===")
    print(summary_logger.get_summary())

if __name__ == "__main__":
    main()