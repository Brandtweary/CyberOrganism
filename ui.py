import tkinter as tk
from tkinter import ttk
import pygame

class UI:
    def __init__(self):
        self.root = None
        self.screen = None
        self.clock = None
        self.font = None
        self.stats_frame = None
        self.canvas = None

    def create_window(self):
        self.root = tk.Tk()
        self.root.title("Simulation")
        self.root.geometry("1920x1080")

        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create sidebar frame
        sidebar_frame = ttk.Frame(main_frame, width=300, style="Sidebar.TFrame")
        sidebar_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Create stats frame (scrollable)
        stats_canvas = tk.Canvas(sidebar_frame, width=280)
        self.stats_frame = ttk.Frame(stats_canvas)
        scrollbar = ttk.Scrollbar(sidebar_frame, orient="vertical", command=stats_canvas.yview)
        stats_canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        stats_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        stats_canvas.create_window((0, 0), window=self.stats_frame, anchor="nw")

        self.stats_frame.bind("<Configure>", lambda e: stats_canvas.configure(scrollregion=stats_canvas.bbox("all")))

        # Create Pygame canvas
        self.canvas = tk.Canvas(main_frame, width=1620, height=1080, bg="black")
        self.canvas.pack(side=tk.RIGHT)

        # Initialize Pygame
        pygame.init()
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.screen = pygame.Surface((1620, 1080))

        # Configure style
        style = ttk.Style()
        style.configure("Sidebar.TFrame", background="gray20")
        style.configure("Stats.TLabel", background="gray20", foreground="lawn green")

        return self.root, self.screen, self.clock, self.font

    def update_stats(self, organism_stats, performance_stats):
        # Clear existing stats
        for widget in self.stats_frame.winfo_children():
            widget.destroy()

        # Add organism stats
        ttk.Label(self.stats_frame, text="Organism Statistics", font=("Any", 14), style="Stats.TLabel").pack(anchor="w")
        for stat in organism_stats:
            ttk.Label(self.stats_frame, text=stat, style="Stats.TLabel").pack(anchor="w")

        # Add a separator
        ttk.Separator(self.stats_frame, orient="horizontal").pack(fill="x", pady=5)

        # Add performance stats
        ttk.Label(self.stats_frame, text="Performance Statistics", font=("Any", 14), style="Stats.TLabel").pack(anchor="w")
        for stat in performance_stats:
            ttk.Label(self.stats_frame, text=stat, style="Stats.TLabel").pack(anchor="w")

    def blit_pygame_surface(self):
        photo_image = tk.PhotoImage(data=pygame.image.tostring(self.screen, 'RGB'))
        self.canvas.create_image(0, 0, image=photo_image, anchor=tk.NW)
        self.canvas.image = photo_image  # Keep a reference to prevent garbage collection

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
        return True

    def run(self):
        self.root.update()
        return self.handle_events()
