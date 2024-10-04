import time
from collections import defaultdict
from typing import Dict, List, Any, Tuple
from contextlib import contextmanager
import threading
from concurrent.futures import ThreadPoolExecutor
import gc


class CustomProfiler:
    def __init__(self):
        self.function_times: Dict[str, List[float]] = defaultdict(list)
        self.section_times: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        self.enabled_functions: Dict[str, bool] = defaultdict(lambda: True)
        self.lock_wait_times: Dict[str, List[float]] = defaultdict(list)
        self.gc_totals: List[int] = []
        self.gc_counts: List[Tuple[int, int, int]] = []
        self.gc_collections: List[Tuple[int, int, int]] = []
        self.last_collections: Tuple[int, int, int] = (0, 0, 0)

    def profile(self, func_name: str):
        def decorator(func):
            def wrapper(*args, **kwargs):
                if not self.enabled_functions[func_name]:
                    return func(*args, **kwargs)
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                self.function_times[func_name].append(end_time - start_time)
                return result
            return wrapper
        return decorator

    @contextmanager
    def profile_section(self, func_name: str, section_name: str):
        start_time = time.perf_counter()
        yield
        end_time = time.perf_counter()
        self.section_times[func_name][section_name].append(end_time - start_time)

    @contextmanager
    def measure_lock_wait(self, func_name: str, lock: threading.Lock):
        start_time = time.perf_counter()
        acquired = lock.acquire(blocking=True)
        end_time = time.perf_counter()
        try:
            self.lock_wait_times[func_name].append(end_time - start_time)
            yield
        finally:
            if acquired:
                lock.release()

    def enable(self, func_name: str):
        self.enabled_functions[func_name] = True

    def disable(self, func_name: str):
        self.enabled_functions[func_name] = False

    def log_gc_count(self):
        count = gc.get_count()
        self.gc_counts.append(count)
        self.gc_totals.append(sum(count))

        stats = gc.get_stats()
        current_collections = tuple(stat['collections'] for stat in stats)
        collections_diff = tuple(current - last for current, last in zip(current_collections, self.last_collections))
        self.gc_collections.append(collections_diff)
        self.last_collections = current_collections

    def get_stats(self, func_name: str) -> Dict[str, Any]:
        times = self.function_times[func_name]
        if not times:
            return {}
        max_time = max(times)
        max_index = times.index(max_time)
        return {
            'avg': sum(times) / len(times),
            'max': max_time,
            'max_index': max_index,
            'calls': len(times)
        }

    def get_section_stats(self, func_name: str, section_name: str) -> Dict[str, Any]:
        times = self.section_times[func_name][section_name]
        if not times:
            return {}
        max_time = max(times)
        max_index = times.index(max_time)
        return {
            'avg': sum(times) / len(times),
            'max': max_time,
            'max_index': max_index,
            'calls': len(times)
        }

    def get_lock_wait_stats(self, func_name: str) -> Dict[str, Any]:
        times = self.lock_wait_times[func_name]
        if not times:
            return {}
        return {
            'avg': sum(times) / len(times),
            'max': max(times),
            'calls': len(times)
        }

    def profile_thread_pool_task(self, func):
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            self.section_times['thread_pool_tasks']['execution_times'].append(execution_time)
            return result
        return wrapper

    def get_performance_stats(self) -> str:
        output = []
        for func_name, times in self.function_times.items():
            if not times:
                continue
            stats = self.get_stats(func_name)
            output.append(f"{func_name} performance:")
            output.append(f"  Avg time: {stats['avg']*1000:.3f}ms")
            output.append(f"  Max time: {stats['max']*1000:.3f}ms (index: {stats['max_index']})")
            output.append(f"  Total calls: {stats['calls']}")
            
            if func_name in self.section_times:
                for section_name, section_times in self.section_times[func_name].items():
                    section_stats = self.get_section_stats(func_name, section_name)
                    output.append(f"  {section_name} section:")
                    output.append(f"    Avg time: {section_stats['avg']*1000:.3f}ms")
                    output.append(f"    Max time: {section_stats['max']*1000:.3f}ms (index: {section_stats['max_index']})")
                    output.append(f"    Total calls: {section_stats['calls']}")
            output.append("")
        
        if 'thread_pool_tasks' in self.section_times:
            execution_times = self.section_times['thread_pool_tasks']['execution_times']
            if execution_times:
                output.append("Thread pool task execution times:")
                output.append(f"  Avg time: {sum(execution_times) / len(execution_times) * 1000:.3f}ms")
                output.append(f"  Max time: {max(execution_times) * 1000:.3f}ms")
                output.append(f"  Total tasks: {len(execution_times)}")
                output.append("")

        for func_name, times in self.lock_wait_times.items():
            if not times:
                continue
            stats = self.get_lock_wait_stats(func_name)
            output.append(f"{func_name} lock wait times:")
            output.append(f"  Avg wait: {stats['avg']*1000:.3f}ms")
            output.append(f"  Max wait: {stats['max']*1000:.3f}ms")
            output.append(f"  Total waits: {stats['calls']}")
            output.append("")

        if self.gc_totals:
            avg_gc = sum(self.gc_totals) / len(self.gc_totals)
            max_gc = max(self.gc_totals)
            max_gc_index = self.gc_totals.index(max_gc)
            
            max_collections = max(self.gc_collections, key=sum)
            max_collections_index = self.gc_collections.index(max_collections)
            
            output.append("GC Stats:")
            output.append(f"  Avg total count: {avg_gc:.2f}")
            output.append(f"  Max total count: {max_gc} (index: {max_gc_index})")
            output.append(f"  Max count breakdown: {self.gc_counts[max_gc_index]}")
            output.append(f"  Max collections: {max_collections} (index: {max_collections_index})")
            output.append("")

        return "\n".join(output)

    def reset_stats(self):
        self.function_times.clear()
        self.section_times.clear()
        self.lock_wait_times.clear()
        self.gc_totals = []
        self.gc_counts = []
        self.gc_collections = []

# Global instance
profiler = CustomProfiler()
profiler.disable("clone_state_snapshot")
profiler.disable("apply_state")
profiler.disable("get_internal_state")
profiler.disable("calculate_nearest_items_vector")
#profiler.disable("update_state")
profiler.disable("queue_learn_conditionally")
#profiler.disable("update_simulation")
profiler.disable("_apply_simulation_state_threaded")
profiler.disable("_apply_simulation_state_sequential")

profiler.disable("_update_snapshot_with_objects_threaded")
profiler.disable("_update_snapshot_with_objects_sequential")

profiler.disable("update_simulation_state")
profiler.disable("simulation_step")
#profiler.disable("select_action")
