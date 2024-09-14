import pygame
import math
from enums import ResizeMode, ContainerType


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
        self.dummy_canvas = pygame.Surface((1, 1))  # Dummy surface for text rendering
        self.ui_elements = self.initialize_ui_elements()
        self.position_ui_elements()

    def estimate_text_dimensions(self, text, font):
        # Estimate width
        text_surface = font.render(text, True, (0, 0, 0))
        width = text_surface.get_width()

        # Estimate height (approximation)
        height = math.ceil(font.get_height() * 1.2)  # Adding 20% for potential line spacing

        return width, height

    def create_ui_element(self, **kwargs):
        default_values = {
            "x": 0,
            "y": 0,
            "height": 0,
            "width": 0,
            "min_height": 0,
            "max_height": float('inf'),
            "content_height": 0,
            "min_width": 0,
            "max_width": float('inf'),
            "content_width": 0,
            "resize_mode": ResizeMode.EXPAND,
            "outer_margin": 0,
            "inner_margin": 0,
            "container_type": ContainerType.VBOX,
            "parent_container_type": None,
            "child_elements": [],
            "background_color": (0, 0, 0, 0),
            "font": self.FONT,
            "font_color": self.NEON_GREEN,
            "text": ""
        }
        
        element = default_values.copy()
        element.update(kwargs)
        return element

    def initialize_ui_elements(self):
        elements = {
            "root": self.create_ui_element(
                min_height=self.HEIGHT,
                max_height=self.HEIGHT,
                min_width=self.WIDTH,
                max_width=self.WIDTH,
                container_type=ContainerType.HBOX,
                child_elements=["left_sidebar", "simulation_area"]
            ),
            "left_sidebar": self.create_ui_element(
                min_height=self.HEIGHT,
                min_width=self.SIDEBAR_WIDTH,
                inner_margin=10,
                container_type=ContainerType.VBOX,
                child_elements=["organism_stats", "performance_stats", "training_metrics"],
                background_color=self.GRAY
            ),
            "simulation_area": self.create_ui_element(
                min_height=self.HEIGHT,
                min_width=self.WIDTH - self.SIDEBAR_WIDTH
            ),
            "organism_stats": self.create_ui_element(
                outer_margin=5,
                inner_margin=5,
                container_type=ContainerType.VBOX,
                child_elements=["organism_stats_title"]
            ),
            "organism_stats_title": self.create_ui_element(
                inner_margin=5,
                container_type=ContainerType.TEXT,
                text="Organism Statistics"
            ),
            "performance_stats": self.create_ui_element(
                outer_margin=5,
                inner_margin=5,
                container_type=ContainerType.VBOX,
                child_elements=["performance_stats_title"]
            ),
            "performance_stats_title": self.create_ui_element(
                inner_margin=5,
                container_type=ContainerType.TEXT,
                text="Performance Statistics"
            ),
            "training_metrics": self.create_ui_element(
                outer_margin=5,
                inner_margin=5,
                container_type=ContainerType.VBOX,
                child_elements=["training_metrics_title"]
            ),
            "training_metrics_title": self.create_ui_element(
                inner_margin=5,
                container_type=ContainerType.TEXT,
                text="Training Metrics"
            )
        }
        return elements

    def set_parent_container_types(self, element_name):
        element = self.ui_elements[element_name]
        for child_name in element['child_elements']:
            child = self.ui_elements[child_name]
            child['parent_container_type'] = element['container_type']
            self.set_parent_container_types(child_name)

    def set_all_content_dimensions(self):
        def calculate_element_dimensions(element_name):
            element = self.ui_elements[element_name]
            
            if element['container_type'] == ContainerType.TEXT:
                text_width, text_height = self.estimate_text_dimensions(element['text'], element['font'])
                element['content_width'] = text_width + 2 * element['inner_margin']
                element['content_height'] = text_height + 2 * element['inner_margin']
            else:  # 'vbox' or 'hbox'
                child_dimensions = [calculate_element_dimensions(child) for child in element['child_elements']]
                
                if element['container_type'] == ContainerType.VBOX:
                    content_width = max([dim[0] for dim in child_dimensions], default=0)
                    content_height = sum(dim[1] + 2 * self.ui_elements[child]['outer_margin'] 
                                         for dim, child in zip(child_dimensions, element['child_elements']))
                else:  # 'hbox'
                    content_width = sum(dim[0] + 2 * self.ui_elements[child]['outer_margin'] 
                                        for dim, child in zip(child_dimensions, element['child_elements']))
                    content_height = max([dim[1] for dim in child_dimensions], default=0)
                
                element['content_width'] = max(content_width + 2 * element['inner_margin'], element['min_width'])
                element['content_height'] = max(content_height + 2 * element['inner_margin'], element['min_height'])
            
            return element['content_width'], element['content_height']

        calculate_element_dimensions('root')

    def position_ui_elements(self):
        def distribute_excess(children, total_excess, dimension):
            expandable_children = [child for child in children if self.ui_elements[child]['resize_mode'] == ResizeMode.EXPAND]
            remaining_excess = total_excess
            
            while remaining_excess > 0 and expandable_children:
                excess_per_child = remaining_excess // len(expandable_children)
                if excess_per_child == 0:
                    break
                
                newly_remaining_excess = remaining_excess
                for child in expandable_children[:]:
                    child_element = self.ui_elements[child]
                    max_additional = child_element[f'max_{dimension}'] - child_element[dimension]
                    additional = min(excess_per_child, max_additional)
                    child_element[dimension] += additional
                    newly_remaining_excess -= additional
                    
                    if child_element[dimension] == child_element[f'max_{dimension}']:
                        expandable_children.remove(child)
                
                remaining_excess = newly_remaining_excess

        def recursive_position(element_name, x, y):
            element = self.ui_elements[element_name]
            element['x'] = x
            element['y'] = y
            
            element['width'] = max(element['width'], element['content_width'])
            element['height'] = max(element['height'], element['content_height'])
            
            available_width = element['width'] - 2 * element['inner_margin']
            available_height = element['height'] - 2 * element['inner_margin']

            if element['container_type'] in [ContainerType.VBOX, ContainerType.HBOX]:
                children = element['child_elements']
                
                if element['container_type'] == ContainerType.VBOX:
                    for child in children:
                        child_element = self.ui_elements[child]
                        child_element['height'] = child_element['content_height']
                        child_element['width'] = min(child_element['max_width'], available_width)
                    
                    total_min_height = sum(self.ui_elements[child]['height'] for child in children)
                    excess_height = max(0, available_height - total_min_height)
                    
                    distribute_excess(children, excess_height, 'height')
                    
                    child_y = y + element['inner_margin']
                    for child in children:
                        child_element = self.ui_elements[child]
                        recursive_position(child, x + element['inner_margin'], child_y)
                        child_y += child_element['height'] + 2 * child_element['outer_margin']
                
                elif element['container_type'] == ContainerType.HBOX:
                    for child in children:
                        child_element = self.ui_elements[child]
                        child_element['width'] = child_element['content_width']
                        child_element['height'] = min(child_element['max_height'], available_height)
                    
                    total_min_width = sum(self.ui_elements[child]['width'] for child in children)
                    excess_width = max(0, available_width - total_min_width)
                    
                    distribute_excess(children, excess_width, 'width')
                    
                    child_x = x + element['inner_margin']
                    for child in children:
                        child_element = self.ui_elements[child]
                        recursive_position(child, child_x, y + element['inner_margin'])
                        child_x += child_element['width'] + 2 * child_element['outer_margin']

        self.set_all_content_dimensions()
        self.set_parent_container_types('root')
        recursive_position('root', 0, 0)

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
    
    def set_text(self, element_name, text):
        element = self.ui_elements[element_name]
        old_content_width, old_content_height = element['content_width'] if 'content_width' in element else 0, element['content_height'] if 'content_height' in element else 0
        element['text'] = text
        
        text_width, text_height = self.estimate_text_dimensions(text, element['font'])
        element['content_width'] = text_width + 2 * element['inner_margin']
        element['content_height'] = text_height + 2 * element['inner_margin']

        # Only reposition if the relevant dimension has changed
        if element['parent_container_type'] == ContainerType.VBOX and element['content_height'] != old_content_height:
            self.position_ui_elements()
        elif element['parent_container_type'] == ContainerType.HBOX and element['content_width'] != old_content_width:
            self.position_ui_elements()
    
    def update_left_sidebar(self, organism_stats, performance_stats, training_metrics):
        old_organism_count = len(self.ui_elements["organism_stats"]["child_elements"]) - 1
        old_performance_count = len(self.ui_elements["performance_stats"]["child_elements"]) - 1
        old_training_metrics_count = len(self.ui_elements["training_metrics"]["child_elements"]) - 1
        
        # Update organism stats
        self.update_stats("organism_stats", organism_stats, old_organism_count)
        
        # Update performance stats
        self.update_stats("performance_stats", performance_stats, old_performance_count)
        
        # Update training metrics
        training_metrics_list = self.format_training_metrics(training_metrics)
        self.update_stats("training_metrics", training_metrics_list, old_training_metrics_count)
        
        # Reposition elements if the number of stats changed
        if (len(organism_stats) != old_organism_count or 
            len(performance_stats) != old_performance_count or
            len(training_metrics_list) != old_training_metrics_count):
            self.position_ui_elements()

    def format_training_metrics(self, training_metrics):
        formatted_metrics = []
        
        # Add combined averages
        combined = training_metrics["combined_averages"]
        formatted_metrics.extend([
            f"Combined Loss: {combined['loss_window_avg']:.4f}",
            f"Combined Q-Value: {combined['current_q_window_avg']:.4f}",
            f"Combined Expected Q: {combined['expected_q_window_avg']:.4f}"
        ])
        
        # Add reward
        formatted_metrics.append(f"Reward: {training_metrics['reward']['reward_window_avg']:.4f}")
        
        # Add metrics for each action
        for action, metrics in training_metrics.items():
            if action not in ["combined_averages", "reward"]:
                formatted_metrics.extend([
                    f"Action {action} Loss: {metrics['loss_window_avg']:.4f}",
                    f"Action {action} Q-Value: {metrics['current_q_window_avg']:.4f}",
                    f"Action {action} Expected Q: {metrics['expected_q_window_avg']:.4f}"
                ])
        
        return formatted_metrics

    def update_stats(self, stats_type, new_stats, old_count):
        self.ui_elements[stats_type]["child_elements"] = [f"{stats_type}_title"]
        for i, stat in enumerate(new_stats):
            stat_block_name = f"{stats_type[:-6]}_stat_{i}"
            if i < old_count:
                # Update existing stat block
                self.set_text(stat_block_name, stat)
            else:
                # Create new stat block using create_ui_element
                self.ui_elements[stat_block_name] = self.create_ui_element(
                    inner_margin=5,
                    container_type="text",
                    text=stat
                )
            self.ui_elements[stats_type]["child_elements"].append(stat_block_name)

    def get_ui_elements(self):
        return self.ui_elements
