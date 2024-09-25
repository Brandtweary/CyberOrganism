from ui import UI
from organism import Organism
from simulation_engine import SimulationEngine
import time
from simulation_state import SimulationState
from PySide6.QtCore import QTimer, QEventLoop, QCoreApplication, QThread, Signal, Slot
import cProfile
import pstats
import io
import os
from collections import deque
from shared_resources import debug
import logging
import traceback

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
    start_time = time.time()

    # Create a Profile object
    pr = cProfile.Profile()
    is_profiling = False

    # Create a deque to store the last two frames' profiling data
    last_frames = deque(maxlen=2)

    def simulation_step():
        nonlocal last_print_time, frame_count, last_fps_calc_time, start_time
        nonlocal frame_times, pr, is_profiling, sim_state, last_frames

        try:
            if debug and not is_profiling and sim_state.frame_count > sim_state.loading_frames + 2:
                is_profiling = True
                logger.info("Warm-up complete. Starting profiling...")

            frame_start = time.time()
            
            if is_profiling:
                if pr.getstats():
                    logger.error("Profiler is already enabled when it shouldn't be. This indicates a bug in the previous frame.")
                    is_profiling = False
                else:
                    pr.enable()

            sim_state.update()
            sim_state.ui.update(sim_state)

            frame_time = time.time() - frame_start
            frame_times.append(frame_time)

            if is_profiling:
                pr.disable()
                
                # Store the profiling data for this frame
                s = io.StringIO()
                ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
                last_frames.append((frame_time, s.getvalue(), ps.stats))

                # Print summary and profile info if frame time exceeds threshold
                if frame_time > 0.030:  # 30ms threshold
                    logger.info(f"\nSlow frame detected: {frame_time*1000:.2f}ms")
                    logger.info("Comparison with previous frame:")
                    print_frame_comparison(last_frames)

                pr.clear()

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
            
            if sim_state.ui.should_exit:
                timer_thread.stop()
                event_loop.quit()

        except Exception as e:
            logger.error(f"Exception in simulation_step: {e}")
            logger.error(traceback.format_exc())
            timer_thread.stop()
            event_loop.quit()
            raise  # Re-raise the exception to be caught in the main thread

    timer_thread.timeout.connect(simulation_step)
    timer_thread.start()

    try:
        event_loop.exec()
    except Exception as e:
        logger.critical(f"Simulation halted due to exception: {e}")
    finally:
        timer_thread.stop()
        timer_thread.wait()  # Wait for the thread to finish

    if not sim_state.ui.should_exit:
        raise RuntimeError("Simulation ended unexpectedly")

    # Print final summary or perform any cleanup tasks here
    print_final_summary(sim_state)

def print_frame_comparison(frames):
    if len(frames) < 2:
        print("Not enough data for comparison")
        return

    prev_frame, slow_frame = frames
    base_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"{'Function':<50} {'Prev Calls':>10} {'Prev Time':>10} {'Prev %':>8} {'Slow Calls':>10} {'Slow Time':>10} {'Slow %':>8} {'Diff':>10}")
    print("=" * 120)

    all_funcs = set(prev_frame[2].keys()) | set(slow_frame[2].keys())
    func_data = []

    for func in all_funcs:
        if base_dir in func[0]:  # Only include functions from your project
            file_name = os.path.basename(func[0])
            func_name = f"{file_name}:{func[2]}"
            
            prev_stats = prev_frame[2].get(func, (0, 0, 0, 0, None))
            slow_stats = slow_frame[2].get(func, (0, 0, 0, 0, None))
            
            prev_calls, prev_time = prev_stats[1], prev_stats[3] * 1000  # Convert to ms
            slow_calls, slow_time = slow_stats[1], slow_stats[3] * 1000  # Convert to ms
            prev_percent = (prev_time / (prev_frame[0] * 1000)) * 100 if prev_frame[0] > 0 else 0
            slow_percent = (slow_time / (slow_frame[0] * 1000)) * 100 if slow_frame[0] > 0 else 0
            time_diff = slow_time - prev_time

            func_data.append((func_name, prev_calls, prev_time, prev_percent, slow_calls, slow_time, slow_percent, time_diff))

    # Sort by slow frame percentage (descending)
    func_data.sort(key=lambda x: x[6], reverse=True)

    for data in func_data:
        print(f"{data[0]:<50} {data[1]:>10} {data[2]:>10.2f} {data[3]:>7.2f}% {data[4]:>10} {data[5]:>10.2f} {data[6]:>7.2f}% {data[7]:>10.2f}")

    print("=" * 120)
    print(f"Total frame time (ms): {prev_frame[0]*1000:>10.2f} {100:>7.2f}% {slow_frame[0]*1000:>10.2f} {100:>7.2f}% {(slow_frame[0] - prev_frame[0])*1000:>10.2f}")
    breakpoint()

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