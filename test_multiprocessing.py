import torch
import torch.multiprocessing as mp
import psutil
import os
import sys
from collections import defaultdict

def worker_process():
    print(f"Worker process ID: {os.getpid()}")
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"Worker Memory Usage: RSS = {memory_info.rss / (1024 * 1024):.2f} MB, VMS = {memory_info.vms / (1024 * 1024):.2f} MB")

    print("\nHigh-level module overview:")
    modules = defaultdict(int)
    for name, module in sys.modules.items():
        if hasattr(module, '__file__') and module.__file__:
                top_level = name.split('.')[0]
                try:
                    modules[top_level] += os.path.getsize(module.__file__)
                except OSError:
                    pass  # Skip if file size cannot be determined

    for name, size in sorted(modules.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{name}: {size / (1024 * 1024):.2f} MB")

def main():
    print(f"Main process ID: {os.getpid()}")
    main_process = psutil.Process(os.getpid())
    memory_info = main_process.memory_info()
    print(f"Main Process Memory Usage: RSS = {memory_info.rss / (1024 * 1024):.2f} MB, VMS = {memory_info.vms / (1024 * 1024):.2f} MB")

    p = mp.Process(target=worker_process)
    p.start()
    p.join()

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()