import uuid

class Item:
    def __init__(self):
        self.id = uuid.uuid4()
        self.color = (255, 255, 255)  # Default color: white
        self.marked_for_deletion = False
        self.expiration = False
        self.expiration_timer = 60  # Default 60 seconds
        self.reward = 0.0
        self.energy = 0.0
        self.nutrition = 0.0

    def consume(self, organism):
        return False  # Base implementation doesn't delete the item

    def apply_state(self, old_state, new_state):
       pass

class Food(Item):
    def __init__(self, expiration_timer=60, energy=1.0, nutrition=0.1, color=(255, 255, 0)):
        super().__init__()
        self.color = color  # Default color: yellow
        self.expiration = True
        self.expiration_timer = expiration_timer
        self.energy = energy
        self.nutrition = nutrition
        self.reward = energy + 5 * nutrition  # Set reward value as energy + 5 * nutrition

    def consume(self, organism):
        organism.energy += self.energy
        organism.nutrition += self.nutrition
        organism.current_reward += self.reward