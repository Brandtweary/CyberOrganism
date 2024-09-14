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
        self.position_ui_elements()

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

    def position_ui_elements(self):
        def recursive_position(element_name, x, y):
            element = self.ui_elements[element_name]
            element['x'] = x
            element['y'] = y
            content_x = x + element["inner_margin"]
            content_y = y + element["inner_margin"]

            if element["container_type"] in ["vbox", "hbox"]:
                child_x, child_y = content_x, content_y
                for child_name in element["child_elements"]:
                    child = self.ui_elements[child_name]
                    recursive_position(child_name, 
                                       child_x + child["outer_margin"], 
                                       child_y + child["outer_margin"])
                    if element["container_type"] == "vbox":
                        child_y += child["height"] + child["outer_margin"] * 2
                    elif element["container_type"] == "hbox":
                        child_x += child["width"] + child["outer_margin"] * 2

        recursive_position("root", 0, 0)

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
        old_organism_count = len(self.ui_elements["organism_stats"]["child_elements"]) - 1
        old_performance_count = len(self.ui_elements["performance_stats"]["child_elements"]) - 1
        
        # Update organism stats
        self.update_stats("organism_stats", organism_stats, old_organism_count)
        
        # Update performance stats
        self.update_stats("performance_stats", performance_stats, old_performance_count)
        
        # Reposition elements if the number of stats changed
        if (len(organism_stats) != old_organism_count or 
            len(performance_stats) != old_performance_count):
            self.position_ui_elements()

    def update_stats(self, stats_type, new_stats, old_count):
        self.ui_elements[stats_type]["child_elements"] = [f"{stats_type}_title"]
        for i, stat in enumerate(new_stats):
            stat_block_name = f"{stats_type[:-6]}_stat_{i}"
            if i < old_count:
                # Update existing stat block
                self.ui_elements[stat_block_name]["text"] = stat
            else:
                # Create new stat block
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
            self.ui_elements[stats_type]["child_elements"].append(stat_block_name)

    def get_ui_elements(self):
        return self.ui_elements
