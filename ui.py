import pygame

class UI:
    def __init__(self):
        pygame.init()
        self.WIDTH, self.HEIGHT = pygame.display.Info().current_w, pygame.display.Info().current_h
        self.TITLE = "Simulation"
        self.BLACK = (0, 0, 0, 255)
        self.GRAY = (50, 50, 50, 255)
        self.NEON_GREEN = (57, 255, 20, 255)
        self.SIDEBAR_WIDTH = 350
        self.FONT = pygame.font.Font(None, 24)
        self.ui_elements = self.initialize_ui_elements()

    def initialize_ui_elements(self):
        elements = {
            "root": {
                "height": self.HEIGHT,
                "width": self.WIDTH,
                "outer_margin": 0,
                "inner_margin": 0,
                "container_type": "hbox",
                "child_elements": ["left_sidebar", "simulation_area"],
                "background_color": (0, 0, 0, 0),
                "font": self.FONT,
                "font_color": self.NEON_GREEN,
                "cacheable": False
            },
            "left_sidebar": {
                "height": self.HEIGHT,
                "width": self.SIDEBAR_WIDTH,
                "outer_margin": 0,
                "inner_margin": 10,
                "container_type": "vbox",
                "child_elements": ["organism_stats", "performance_stats"],
                "background_color": self.GRAY,
                "font": self.FONT,
                "font_color": self.NEON_GREEN,
                "cacheable": True
            },
            "simulation_area": {
                "height": self.HEIGHT,
                "width": self.WIDTH - self.SIDEBAR_WIDTH,
                "outer_margin": 0,
                "inner_margin": 0,
                "container_type": "vbox",
                "child_elements": [],
                "background_color": (0, 0, 0, 0),
                "font": self.FONT,
                "font_color": self.NEON_GREEN,
                "cacheable": False
            },
            "organism_stats": {
                "height": self.HEIGHT // 2,
                "width": self.SIDEBAR_WIDTH,
                "outer_margin": 5,
                "inner_margin": 5,
                "container_type": "vbox",
                "child_elements": ["organism_stats_title"],
                "background_color": (0, 0, 0, 0),
                "font": self.FONT,
                "font_color": self.NEON_GREEN,
                "cacheable": False
            },
            "organism_stats_title": {
                "height": 30,
                "width": self.SIDEBAR_WIDTH - 20,
                "outer_margin": 0,
                "inner_margin": 5,
                "container_type": "text",
                "text": "Organism Statistics",
                "background_color": (0, 0, 0, 0),
                "font": self.FONT,
                "font_color": self.NEON_GREEN,
                "cacheable": False
            },
            "performance_stats": {
                "height": self.HEIGHT // 2,
                "width": self.SIDEBAR_WIDTH,
                "outer_margin": 5,
                "inner_margin": 5,
                "container_type": "vbox",
                "child_elements": ["performance_stats_title"],
                "background_color": (0, 0, 0, 0),
                "font": self.FONT,
                "font_color": self.NEON_GREEN,
                "cacheable": False
            },
            "performance_stats_title": {
                "height": 30,
                "width": self.SIDEBAR_WIDTH - 20,
                "outer_margin": 0,
                "inner_margin": 5,
                "container_type": "text",
                "text": "Performance Statistics",
                "background_color": (0, 0, 0, 0),
                "font": self.FONT,
                "font_color": self.NEON_GREEN,
                "cacheable": False
            }
        }
        return elements

    def create_window(self):
        display = pygame.display.set_mode((self.WIDTH, self.HEIGHT), pygame.FULLSCREEN)
        pygame.display.set_caption(self.TITLE)
        clock = pygame.time.Clock()
        return display, clock

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
        return True

    def update_left_sidebar(self, organism_stats, performance_stats):
        # Clear existing stat blocks
        self.ui_elements["organism_stats"]["child_elements"] = ["organism_stats_title"]
        self.ui_elements["performance_stats"]["child_elements"] = ["performance_stats_title"]

        # Add new organism stat blocks
        for i, stat in enumerate(organism_stats):
            stat_block_name = f"organism_stat_{i}"
            self.ui_elements[stat_block_name] = {
                "height": 25,
                "width": self.SIDEBAR_WIDTH - 20,
                "outer_margin": 0,
                "inner_margin": 5,
                "container_type": "text",
                "text": stat,
                "background_color": (0, 0, 0, 0),
                "font": self.FONT,
                "font_color": self.NEON_GREEN,
                "cacheable": False
            }
            self.ui_elements["organism_stats"]["child_elements"].append(stat_block_name)

        # Add new performance stat blocks
        for i, stat in enumerate(performance_stats):
            stat_block_name = f"performance_stat_{i}"
            self.ui_elements[stat_block_name] = {
                "height": 25,
                "width": self.SIDEBAR_WIDTH - 20,
                "outer_margin": 0,
                "inner_margin": 5,
                "container_type": "text",
                "text": stat,
                "background_color": (0, 0, 0, 0),
                "font": self.FONT,
                "font_color": self.NEON_GREEN,
                "cacheable": False

            }
            self.ui_elements["performance_stats"]["child_elements"].append(stat_block_name)

    def get_ui_elements(self):
        return self.ui_elements
