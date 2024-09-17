import dearpygui.dearpygui as dpg
from drawing import draw_simulation
from screeninfo import get_monitors

class UI:
    def __init__(self):
        self.monitor = get_monitors()[0]  # Get the primary monitor
        self.WIDTH, self.HEIGHT = self.monitor.width, self.monitor.height
        self.SIDEBAR_WIDTH = 350
        self.TITLE = "CyberOrganism"
        self.should_exit = False

        dpg.create_context()
        self.setup_ui()

    def setup_ui(self):
        # Load custom font
        with dpg.font_registry():
            default_font = dpg.add_font("fonts/Roboto_Mono/static/RobotoMono-Regular.ttf", 22, pixel_snapH=True)        
        
        # Set default font
        dpg.bind_font(default_font)

        # Create the main viewport
        dpg.create_viewport(title=self.TITLE, width=self.WIDTH, height=self.HEIGHT)

        # Create the main window
        with dpg.window(label="Main", tag="main_window", autosize=True):
            with dpg.window(label="Simulation", no_title_bar=True, no_move=True, no_resize=True, tag="sim_window"):
                dpg.add_drawlist(width=self.WIDTH, height=self.HEIGHT, tag="sim_area")

            with dpg.window(label="Sidebar", no_title_bar=True, no_move=True, no_resize=True, tag="sidebar_window"):
                with dpg.child_window(width=self.SIDEBAR_WIDTH, height=self.HEIGHT, tag="sidebar"):
                    dpg.add_text("Simulation Stats", color=(0, 255, 0))
                    
                    with dpg.collapsing_header(label="Organism Statistics", default_open=True):
                        dpg.add_text("", tag="organism_stats")
                    
                    with dpg.collapsing_header(label="Performance Statistics", default_open=True):
                        dpg.add_text("", tag="performance_stats")
                    
                    with dpg.collapsing_header(label="Training Metrics", default_open=True):
                        dpg.add_text("", tag="training_metrics")

        # Set the primary window
        dpg.set_primary_window("main_window", True)

        # Set the sidebar background color
        with dpg.theme() as sidebar_theme:
            with dpg.theme_component(dpg.mvChildWindow):
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (47, 79, 79))  # Slate gray

        dpg.bind_item_theme("sidebar", sidebar_theme)

        # Set the main window background color
        with dpg.theme() as main_window_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (0, 0, 0, 255))  # Black background
                dpg.add_theme_color(dpg.mvThemeCol_Border, (0, 0, 0, 0))  # Transparent border

        dpg.bind_item_theme("main_window", main_window_theme)

        # Set the global font color and font
        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_Text, (0, 255, 0))  # Neon green

        dpg.bind_theme(global_theme)

        # Set up key handler
        with dpg.handler_registry():
            dpg.add_key_release_handler(key=dpg.mvKey_Escape, callback=self.exit_callback)

        # Configure viewport
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.toggle_viewport_fullscreen()
    
    def exit_callback(self):
        self.should_exit = True

    def update_left_sidebar(self, organism_stats, performance_stats, training_metrics):
        dpg.set_value("organism_stats", self.format_stats(organism_stats))
        dpg.set_value("performance_stats", self.format_stats(performance_stats))
        dpg.set_value("training_metrics", self.format_training_metrics(training_metrics))

    def format_stats(self, stats):
        return "\n".join(stats)

    def format_training_metrics(self, training_metrics):
        formatted_metrics = []
        combined = training_metrics["combined_averages"]
        formatted_metrics.extend([
            f"Combined Loss: {combined['loss_window_avg']:.4f}",
            f"Combined Q-Value: {combined['current_q_window_avg']:.4f}",
            f"Combined Expected Q: {combined['expected_q_window_avg']:.4f}",
            f"Reward: {training_metrics['reward']['reward_window_avg']:.4f}"
        ])
        for action, metrics in training_metrics.items():
            if action not in ["combined_averages", "reward"]:
                formatted_metrics.extend([
                    f"Action {action} Loss: {metrics['loss_window_avg']:.4f}",
                    f"Action {action} Q-Value: {metrics['current_q_window_avg']:.4f}",
                    f"Action {action} Expected Q: {metrics['expected_q_window_avg']:.4f}"
                ])
        return "\n".join(formatted_metrics)

    def update(self, sim_state):
        with dpg.mutex():
            dpg.delete_item("sim_area", children_only=True)
            draw_simulation("sim_area", sim_state)

        dpg.render_dearpygui_frame()
        
    def should_quit(self):
        return self.should_exit or not dpg.is_dearpygui_running()

    def cleanup(self):
        dpg.destroy_context()

    def get_display_framerate(self):
        return dpg.get_frame_rate()
    
    def get_mouse_position(self):
        return dpg.get_mouse_pos()
    
    def get_viewport_dimensions(self):
        return dpg.get_viewport_width(), dpg.get_viewport_height()
