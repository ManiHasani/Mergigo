"""
Drop & Merge Physics Game

A physics-based puzzle game where players drop numbered circles that merge 
when identical values collide. Features realistic physics simulation, 
particle effects, and progressive difficulty scaling.

Author: Mani Hasani
Version: 1.2.0
"""

import pygame
import pygame.gfxdraw
import math
import random
import sys
import os
from typing import Tuple, List, Optional

# Initialize Pygame
pygame.init()

# Game Configuration Constants
class Config:
    """Central configuration management for game parameters."""

    # Display settings
    SCREEN_WIDTH = 500
    SCREEN_HEIGHT = 700
    FPS = 120

    # Physics parameters
    GRAVITY = 0.3
    FRICTION = 0.995
    BOUNCE_DAMPING = 0.6
    BOUNCE_STOP_THRESHOLD = 0.3
    GROUND_FRICTION = 0.78
    MERGE_THRESHOLD = 0.3
    MAX_VELOCITY = 8.0

    # Game boundaries
    PLAY_AREA_LEFT = 75
    PLAY_AREA_RIGHT = SCREEN_WIDTH - 75
    PLAY_AREA_TOP = 120
    PLAY_AREA_BOTTOM = SCREEN_HEIGHT - 60

    # Visual settings
    PARTICLE_LIFETIME = 60
    MERGE_ANIMATION_DURATION = 25
    SCORE_PULSE_DURATION = 8


class Colors:
    """Centralized color definitions for consistent theming."""

    # Basic colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    DARK_GRAY = (50, 50, 50)
    LIGHT_GRAY = (200, 200, 200)
    GOLD = (255, 215, 0)
    GREEN = (50, 200, 50)
    RED = (200, 50, 50)
    BLUE = (50, 50, 200)

    # Value-based color schemes with gradients
    CIRCLE_COLORS = {
        2: [(255, 100, 100), (255, 150, 150)],
        4: [(100, 255, 100), (150, 255, 150)],
        8: [(100, 100, 255), (150, 150, 255)],
        16: [(255, 255, 100), (255, 255, 180)],
        32: [(255, 100, 255), (255, 150, 255)],
        64: [(100, 255, 255), (150, 255, 255)],
        128: [(255, 165, 0), (255, 200, 100)],
        256: [(128, 128, 128), (180, 180, 180)],
        512: [(255, 50, 50), (255, 100, 100)],
        1024: [(50, 255, 50), (100, 255, 100)],
    }

    @classmethod
    def get_circle_colors(cls, value: int) -> List[Tuple[int, int, int]]:
        """Get color scheme for a given circle value."""
        return cls.CIRCLE_COLORS.get(value, [(255, 255, 255), (200, 200, 200)])


class UIButton:
    """
    Interactive button component with hover effects and state management.
    Provides consistent user interface elements throughout the application.
    """

    def __init__(self, x: int, y: int, width: int, height: int, 
                 text: str, color: Tuple[int, int, int], 
                 hover_color: Tuple[int, int, int], 
                 text_color: Tuple[int, int, int] = Colors.WHITE):
        """
        Initialize button with position and appearance properties.

        Args:
            x: Horizontal position
            y: Vertical position
            width: Button width in pixels
            height: Button height in pixels
            text: Display text
            color: Default background color
            hover_color: Hover state background color
            text_color: Text color
        """
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.font = pygame.font.Font(None, 24)
        self.is_hovered = False
        self.is_pressed = False

    def handle_event(self, event: pygame.event.Event) -> bool:
        """
        Process mouse events and update button state.

        Args:
            event: Pygame event to process

        Returns:
            True if button was clicked, False otherwise
        """
        mouse_pos = pygame.mouse.get_pos()

        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(mouse_pos)

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and self.rect.collidepoint(mouse_pos):
                self.is_pressed = True
                return True

        elif event.type == pygame.MOUSEBUTTONUP:
            self.is_pressed = False

        return False

    def render(self, screen: pygame.Surface) -> None:
        """
        Draw button with current state visual effects.

        Args:
            screen: Target surface for rendering
        """
        # Determine button color based on state
        current_color = self.hover_color if self.is_hovered else self.color
        if self.is_pressed:
            current_color = tuple(max(0, c - 30) for c in current_color)

        # Draw button background with rounded corners
        pygame.draw.rect(screen, current_color, self.rect, border_radius=8)
        pygame.draw.rect(screen, Colors.BLACK, self.rect, 2, border_radius=8)

        # Render centered text
        text_surface = self.font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)


class ParticleEffect:
    """
    Individual particle for visual effects system.
    Handles physics simulation and rendering for particle animations.
    """

    def __init__(self, x: float, y: float, color: Tuple[int, int, int], 
                 velocity: Tuple[float, float] = (0.0, 0.0)):
        """
        Initialize particle with physics and visual properties.

        Args:
            x: Initial X position
            y: Initial Y position
            color: RGB color tuple
            velocity: Initial velocity as (vx, vy)
        """
        self.position = pygame.math.Vector2(x, y)
        self.velocity = pygame.math.Vector2(velocity)
        self.color = color
        self.life = Config.PARTICLE_LIFETIME
        self.max_life = Config.PARTICLE_LIFETIME
        self.size = random.randint(2, 4)

    def update(self) -> bool:
        """
        Update particle physics and lifetime.

        Returns:
            True if particle is still alive, False if expired
        """
        # Apply physics
        self.position += self.velocity
        self.velocity.y += 0.15  # Gravity effect
        self.life -= 1

        return self.life > 0

    def render(self, screen: pygame.Surface) -> None:
        """
        Draw particle with alpha blending based on remaining life.

        Args:
            screen: Target surface for rendering
        """
        if self.life <= 0:
            return

        # Calculate alpha based on remaining life
        alpha = int(255 * (self.life / self.max_life))
        color_with_alpha = (*self.color, alpha)

        # Create surface with alpha blending
        particle_surface = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)
        pygame.draw.circle(particle_surface, color_with_alpha, 
                         (self.size, self.size), self.size)

        screen.blit(particle_surface, 
                   (self.position.x - self.size, self.position.y - self.size))


class GameCircle:
    """
    Primary game object representing numbered circles with physics simulation.
    Handles movement, collision detection, merging mechanics, and visual rendering.
    """

    def __init__(self, x: float, y: float, radius: float, value: int):
        """
        Initialize circle with position, physics, and visual properties.

        Args:
            x: Initial X coordinate
            y: Initial Y coordinate
            radius: Circle radius in pixels
            value: Numeric value displayed on circle
        """
        self.position = pygame.math.Vector2(x, y)
        self.velocity = pygame.math.Vector2(0.0, 0.0)
        self.radius = radius
        self.value = value
        self.color_scheme = Colors.get_circle_colors(value)

        # State tracking
        self.is_merged = False
        self.on_ground = False

        # Visual effects
        self.animation_scale = 1.0
        self.merge_animation_timer = 0
        self.glow_intensity = 0.0
        self.bounce_scale = 1.0

        # Physics properties
        self.collision_radius = radius - 1.0
        self.mass = radius * 0.1
        self.separation_force = pygame.math.Vector2(0, 0)

        # Ground interaction tracking
        self.ground_bounce_count = 0
        self.last_y_velocity = 0.0
        self.ground_contact_time = 0

    def update(self) -> None:
        """Execute complete physics and animation update cycle."""
        self._update_animations()
        self._apply_physics()
        self._handle_boundary_collisions()
        self._update_ground_state()

    def _update_animations(self) -> None:
        """Update visual animation states."""
        self.last_y_velocity = self.velocity.y

        # Merge animation
        if self.merge_animation_timer > 0:
            self.animation_scale = 1.0 + math.sin(self.merge_animation_timer * 0.3) * 0.15
            self.merge_animation_timer -= 1
            self.glow_intensity = max(0, self.merge_animation_timer / 30.0)
        else:
            self.animation_scale = 1.0
            self.glow_intensity = 0.0

        # Bounce animation interpolation
        if self.bounce_scale != 1.0:
            self.bounce_scale += (1.0 - self.bounce_scale) * 0.15
            if abs(self.bounce_scale - 1.0) < 0.01:
                self.bounce_scale = 1.0

    def _apply_physics(self) -> None:
        """Apply physics simulation including gravity and constraints."""
        # Apply separation forces from collisions
        self.velocity += self.separation_force
        self.separation_force.update(0, 0)

        # Apply gravity
        self.velocity.y += Config.GRAVITY

        # Velocity constraints
        if self.velocity.length() > Config.MAX_VELOCITY:
            self.velocity.scale_to_length(Config.MAX_VELOCITY)

        # Update position
        self.position += self.velocity

        # Air resistance
        self.velocity.x *= Config.FRICTION

    def _handle_boundary_collisions(self) -> None:
        """Handle collisions with game area boundaries."""
        # Left wall collision
        if self.position.x - self.collision_radius <= Config.PLAY_AREA_LEFT:
            self.position.x = Config.PLAY_AREA_LEFT + self.collision_radius
            self.velocity.x = -self.velocity.x * Config.BOUNCE_DAMPING

        # Right wall collision
        if self.position.x + self.collision_radius >= Config.PLAY_AREA_RIGHT:
            self.position.x = Config.PLAY_AREA_RIGHT - self.collision_radius
            self.velocity.x = -self.velocity.x * Config.BOUNCE_DAMPING

        # Top boundary collision
        if self.position.y - self.collision_radius <= Config.PLAY_AREA_TOP:
            self.position.y = Config.PLAY_AREA_TOP + self.collision_radius
            self.velocity.y = abs(self.velocity.y) * Config.BOUNCE_DAMPING * 0.3

        # Ground collision with bounce mechanics
        if self.position.y + self.collision_radius >= Config.PLAY_AREA_BOTTOM:
            self.position.y = Config.PLAY_AREA_BOTTOM - self.collision_radius
            self._handle_ground_bounce()

    def _handle_ground_bounce(self) -> None:
        """Process ground collision with realistic bouncing."""
        if self.last_y_velocity > 1.0:
            bounce_strength = min(self.last_y_velocity * Config.BOUNCE_DAMPING, 6.0)
            bounce_reduction = max(0.3, 1.0 - (self.ground_bounce_count * 0.15))

            self.velocity.y = -bounce_strength * bounce_reduction
            self.bounce_scale = 1.0 + (bounce_strength * 0.04)
            self.ground_bounce_count += 1
            self.velocity.x *= Config.GROUND_FRICTION
        else:
            self.velocity.y = 0
            self.ground_bounce_count = 0

    def _update_ground_state(self) -> None:
        """Update ground contact state and apply stability corrections."""
        # Reset bounce counter when airborne
        if self.position.y + self.collision_radius < Config.PLAY_AREA_BOTTOM - 5:
            self.ground_bounce_count = 0
            self.on_ground = False
            self.ground_contact_time = 0

        # Ground contact detection
        self.on_ground = (self.position.y + self.collision_radius >= Config.PLAY_AREA_BOTTOM - 2)
        if self.on_ground:
            self.ground_contact_time += 1

        # Velocity dampening for stability
        if abs(self.velocity.y) < Config.BOUNCE_STOP_THRESHOLD and self.on_ground:
            self.velocity.y = 0

        # Horizontal movement dampening
        threshold = 0.05 if self.on_ground else 0.02
        if abs(self.velocity.x) < threshold:
            self.velocity.x = 0

    def render(self, screen: pygame.Surface) -> None:
        """
        Render circle with gradient effects, glow, and value display.

        Args:
            screen: Target surface for rendering
        """
        draw_radius = int(self.radius * self.animation_scale * self.bounce_scale)
        center = (int(self.position.x), int(self.position.y))

        # Draw gradient fill
        self._draw_gradient(screen, center, draw_radius)

        # Draw glow effect during animations
        if self.glow_intensity > 0:
            self._draw_glow_effect(screen, center, draw_radius)

        # Draw border highlights
        pygame.gfxdraw.aacircle(screen, center[0], center[1], draw_radius, (240, 240, 240))
        pygame.gfxdraw.aacircle(screen, center[0], center[1], draw_radius - 1, (90, 90, 90))

        # Render value text
        self._draw_value_text(screen, center, draw_radius)

    def _draw_gradient(self, screen: pygame.Surface, center: Tuple[int, int], 
                      draw_radius: int) -> None:
        """Draw gradient fill effect."""
        for i in range(draw_radius, 0, -2):
            factor = i / draw_radius
            color = [
                int(self.color_scheme[1][j] + (self.color_scheme[0][j] - self.color_scheme[1][j]) * factor)
                for j in range(3)
            ]
            pygame.gfxdraw.aacircle(screen, center[0], center[1], i, color)
            pygame.gfxdraw.filled_circle(screen, center[0], center[1], i, color)

    def _draw_glow_effect(self, screen: pygame.Surface, center: Tuple[int, int], 
                         draw_radius: int) -> None:
        """Draw glow effect during animations."""
        glow_radius = draw_radius + int(8 * self.glow_intensity)
        glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        glow_color = (*self.color_scheme[0], int(60 * self.glow_intensity))
        pygame.draw.circle(glow_surface, glow_color, (glow_radius, glow_radius), glow_radius)
        screen.blit(glow_surface, (center[0] - glow_radius, center[1] - glow_radius))

    def _draw_value_text(self, screen: pygame.Surface, center: Tuple[int, int], 
                        draw_radius: int) -> None:
        """Draw numeric value text on circle."""
        font_size = max(16, min(28, draw_radius // 2))
        font = pygame.font.Font(None, font_size)
        text_surface = font.render(str(self.value), True, Colors.BLACK)
        text_rect = text_surface.get_rect(center=center)
        screen.blit(text_surface, text_rect)

    def get_distance_to(self, other: 'GameCircle') -> float:
        """Calculate distance to another circle."""
        return self.position.distance_to(other.position)

    def is_colliding_with(self, other: 'GameCircle') -> bool:
        """Check collision with another circle."""
        distance = self.get_distance_to(other)
        return distance < (self.collision_radius + other.collision_radius)

    def resolve_collision_with(self, other: 'GameCircle') -> None:
        """
        Resolve physics collision with another circle.

        Args:
            other: Circle to resolve collision with
        """
        distance = self.get_distance_to(other)
        min_distance = self.collision_radius + other.collision_radius

        if distance < min_distance and distance > 0:
            # Calculate collision direction
            direction = (self.position - other.position).normalize()
            overlap = min_distance - distance

            # Mass-based separation
            total_mass = self.mass + other.mass
            self_separation = overlap * (other.mass / total_mass) * 0.52
            other_separation = overlap * (self.mass / total_mass) * 0.52

            # Apply position correction
            self.position += direction * self_separation
            other.position -= direction * other_separation

            # Apply separation forces
            separation_force = overlap * 0.1
            self.separation_force += direction * separation_force * (other.mass / total_mass)
            other.separation_force -= direction * separation_force * (self.mass / total_mass)

            # Velocity collision response
            relative_velocity = self.velocity - other.velocity
            velocity_along_normal = relative_velocity.dot(direction)

            if velocity_along_normal < 0:
                impulse = -1.6 * velocity_along_normal / total_mass
                self.velocity += direction * impulse * other.mass
                other.velocity -= direction * impulse * self.mass

    def can_merge_with(self, other: 'GameCircle') -> bool:
        """Check if this circle can merge with another."""
        return (self.value == other.value and 
                not self.is_merged and not other.is_merged and
                self.value < 1024)

    def create_merged_circle(self, other: 'GameCircle') -> 'GameCircle':
        """Create a new merged circle from this circle and another."""
        merge_position = (self.position + other.position) / 2
        new_value = self.value * 2
        new_radius = CircleManager.get_radius_for_value(new_value)

        merged_circle = GameCircle(merge_position.x, merge_position.y, new_radius, new_value)
        merged_circle.is_merged = True
        merged_circle.merge_animation_timer = Config.MERGE_ANIMATION_DURATION
        merged_circle.velocity = (self.velocity + other.velocity) * 0.25

        return merged_circle


class CircleManager:
    """Utility class for circle-related calculations and generation."""

    @staticmethod
    def get_radius_for_value(value: int) -> int:
        """
        Calculate appropriate radius for a given value.

        Args:
            value: Circle value

        Returns:
            Calculated radius in pixels
        """
        base_radius = 20
        scale_factor = math.log2(value)
        return int(base_radius + scale_factor * 8)

    @staticmethod
    def generate_next_value() -> int:
        """
        Generate next circle value using weighted probability.

        Returns:
            Selected value for next circle
        """
        values = [2, 4, 8, 16]
        weights = [45, 30, 20, 5]
        return random.choices(values, weights=weights)[0]


class ScoreManager:
    """Handles score persistence and high score tracking (file next to exe/py)."""

    def __init__(self, filename: str = "highscore.dat"):
        """
        Create ScoreManager that stores highscore in `filename` placed next to exe/.py.
        If file doesn't exist, it will be created with 0.
        """
        base_dir = get_base_dir()
        self.filename = os.path.join(base_dir, filename)

        # try to ensure the file exists (create with 0 if missing)
        try:
            if not os.path.exists(self.filename):
                with open(self.filename, "w", encoding="utf-8") as f:
                    f.write("0")
        except Exception:
            # if we can't create the file (e.g. permission error), fall back to memory-only mode
            # keep filename as None to indicate persistence not available
            self.filename = None

        self.high_score = self._load_high_score()

    def _load_high_score(self) -> int:
        """Read integer high score from file (or return 0)."""
        if not self.filename:
            return 0
        try:
            with open(self.filename, "r", encoding="utf-8") as f:
                content = f.read().strip()
                return int(content) if content else 0
        except Exception:
            return 0

    def update_high_score(self, current_score: int) -> bool:
        """
        Update high score if current_score is greater.
        Returns True if high score was updated (even if write to disk failed).
        """
        if current_score > self.high_score:
            self.high_score = current_score
            # try to persist to disk (if possible)
            if self.filename:
                try:
                    with open(self.filename, "w", encoding="utf-8") as f:
                        f.write(str(self.high_score))
                except Exception:
                    # write failed (permissions, locked file, etc.) — we still keep high_score in memory
                    pass
            return True
        return False


class GameEngine:
    """
    Core game engine managing all systems and game loop.
    Coordinates physics, rendering, input handling, and game state.
    """

    def __init__(self):
        """Initialize all game systems and components."""
        self._setup_display()
        self._initialize_game_state()
        self._create_ui_elements()
        self._setup_fonts()

    def _setup_display(self) -> None:
        """Initialize display system and window."""
        self.screen = pygame.display.set_mode((Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT))
        pygame.display.set_caption("mergigo")
        self.clock = pygame.time.Clock()

    def _initialize_game_state(self) -> None:
        """Initialize game state variables."""
        self.circles: List[GameCircle] = []
        self.particles: List[ParticleEffect] = []
        self.score = 0
        self.game_over = False
        self.drop_x = Config.SCREEN_WIDTH // 2

        self.next_value = CircleManager.generate_next_value()
        self.next_radius = CircleManager.get_radius_for_value(self.next_value)

        self.score_manager = ScoreManager()
        self.new_high_score_achieved = False

        # Visual effect variables
        self.background_offset = 0.0
        self.score_pulse_timer = 0

    def _create_ui_elements(self) -> None:
        """Create user interface buttons."""
        # Main game buttons
        self.restart_button = UIButton(Config.SCREEN_WIDTH - 180, 10, 80, 35, 
                                     "Restart", Colors.GREEN, (80, 230, 80))
        self.exit_button = UIButton(Config.SCREEN_WIDTH - 90, 10, 80, 35,
                                  "Exit", Colors.RED, (230, 80, 80))

        # Game over screen buttons
        self.game_over_restart_btn = UIButton(Config.SCREEN_WIDTH // 2 - 100, 
                                            Config.SCREEN_HEIGHT // 2 + 90, 
                                            90, 40, "Restart", Colors.GREEN, (80, 230, 80))
        self.game_over_exit_btn = UIButton(Config.SCREEN_WIDTH // 2 + 10,
                                         Config.SCREEN_HEIGHT // 2 + 90,
                                         90, 40, "Exit", Colors.RED, (230, 80, 80))

    def _setup_fonts(self) -> None:
        """Initialize font resources."""
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

    def drop_circle(self) -> None:
        """Create and drop a new circle at current position."""
        if self.game_over:
            return

        radius = CircleManager.get_radius_for_value(self.next_value)
        safe_x = max(Config.PLAY_AREA_LEFT + radius, 
                    min(Config.PLAY_AREA_RIGHT - radius, self.drop_x))

        new_circle = GameCircle(safe_x, Config.PLAY_AREA_TOP + radius, radius, self.next_value)
        new_circle.velocity.y = 0.5
        self.circles.append(new_circle)

        self.next_value = CircleManager.generate_next_value()
        self.next_radius = CircleManager.get_radius_for_value(self.next_value)

    def update_drop_position(self) -> None:
        """Update drop position based on mouse with smooth interpolation."""
        if self.game_over:
            return

        mouse_x, _ = pygame.mouse.get_pos()
        target_x = max(Config.PLAY_AREA_LEFT + self.next_radius,
                      min(Config.PLAY_AREA_RIGHT - self.next_radius, mouse_x))
        self.drop_x += (target_x - self.drop_x) * 0.12

    def create_particle_burst(self, x: float, y: float, 
                            color: Tuple[int, int, int], count: int = 10) -> None:
        """Create particle explosion effect at specified location."""
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 5)
            velocity = (math.cos(angle) * speed, math.sin(angle) * speed - 1)
            self.particles.append(ParticleEffect(x, y, color, velocity))

    def update_physics(self) -> None:
        """Update complete physics simulation."""
        # Update all circles
        for circle in self.circles:
            circle.update()
            circle.is_merged = False

        # Update particles
        self.particles = [p for p in self.particles if p.update()]

        # Multi-pass collision detection
        self._handle_collisions()

    def _handle_collisions(self) -> None:
        """Handle collision detection and resolution."""
        for collision_pass in range(3):
            circles_to_remove = []
            circles_to_add = []

            for i in range(len(self.circles)):
                if i >= len(self.circles):  # List modified during iteration
                    break

                for j in range(i + 1, len(self.circles)):
                    if j >= len(self.circles):
                        break

                    circle1 = self.circles[i]
                    circle2 = self.circles[j]

                    if circle1.is_colliding_with(circle2):
                        if (collision_pass == 0 and circle1.can_merge_with(circle2)):
                            # Handle merge
                            merged_circle = circle1.create_merged_circle(circle2)

                            # Create particle effect
                            merge_pos = (circle1.position + circle2.position) / 2
                            self.create_particle_burst(merge_pos.x, merge_pos.y, 
                                                     circle1.color_scheme[0])

                            # Update score
                            score_bonus = merged_circle.value * (1 + math.log2(merged_circle.value) * 0.1)
                            self.score += int(score_bonus)
                            self.score_pulse_timer = Config.SCORE_PULSE_DURATION

                            # Schedule circle replacement
                            if circle1 not in circles_to_remove:
                                circles_to_remove.append(circle1)
                            if circle2 not in circles_to_remove:
                                circles_to_remove.append(circle2)
                            circles_to_add.append(merged_circle)
                            break
                        else:
                            # Handle regular collision
                            circle1.resolve_collision_with(circle2)

            # Apply circle changes
            for circle in circles_to_remove:
                if circle in self.circles:
                    self.circles.remove(circle)
            self.circles.extend(circles_to_add)

    def check_game_over_conditions(self) -> None:
        """Evaluate game over conditions."""
        danger_threshold = Config.PLAY_AREA_TOP + 100
        circles_in_danger = sum(1 for circle in self.circles 
                               if circle.position.y - circle.collision_radius <= danger_threshold)

        if circles_in_danger > 3:
            self.game_over = True
            self.new_high_score_achieved = self.score_manager.update_high_score(self.score)

    def render_background(self) -> None:
        """Render animated background with gradient and pattern."""
        self.background_offset += 0.3
        if self.background_offset > 20:
            self.background_offset = 0

        # Gradient background
        for y in range(Config.SCREEN_HEIGHT):
            color_factor = y / Config.SCREEN_HEIGHT
            color = (
                int(240 + color_factor * 15),
                int(248 + color_factor * 7),
                255
            )
            pygame.draw.line(self.screen, color, (0, y), (Config.SCREEN_WIDTH, y))

        # Animated dot pattern
        for x in range(-20, Config.SCREEN_WIDTH + 20, 40):
            for y in range(-20, Config.SCREEN_HEIGHT + 20, 40):
                dot_pos = (int(x + self.background_offset), int(y + self.background_offset))
                pygame.draw.circle(self.screen, (230, 230, 240), dot_pos, 1)

    def render_game_area(self) -> None:
        """Render game area boundaries and danger line."""
        # Game area border
        area_rect = pygame.Rect(Config.PLAY_AREA_LEFT - 5, Config.PLAY_AREA_TOP - 5,
                               Config.PLAY_AREA_RIGHT - Config.PLAY_AREA_LEFT + 10,
                               Config.PLAY_AREA_BOTTOM - Config.PLAY_AREA_TOP + 10)
        pygame.draw.rect(self.screen, Colors.DARK_GRAY, area_rect, 3)
        pygame.draw.rect(self.screen, Colors.LIGHT_GRAY, area_rect, 1)

        # Animated danger line
        danger_alpha = int(100 + 50 * math.sin(pygame.time.get_ticks() * 0.008))
        danger_surface = pygame.Surface((Config.PLAY_AREA_RIGHT - Config.PLAY_AREA_LEFT, 2), 
                                       pygame.SRCALPHA)
        danger_surface.fill((255, 0, 0, danger_alpha))
        self.screen.blit(danger_surface, (Config.PLAY_AREA_LEFT, Config.PLAY_AREA_TOP + 100))

    def render_preview_circle(self) -> None:
        """Render next circle preview and drop line."""
        preview_y = 70
        preview_circle = GameCircle(self.drop_x, preview_y, self.next_radius, self.next_value)
        preview_circle.glow_intensity = 0.25
        preview_circle.render(self.screen)

        # Drop trajectory line
        pygame.draw.line(self.screen, (200, 200, 200, 100),
                        (self.drop_x, preview_y + self.next_radius + 5),
                        (self.drop_x, Config.PLAY_AREA_TOP), 2)

    def render_ui(self) -> None:
        """Render user interface elements."""
        # Score with pulse effect
        if self.score_pulse_timer > 0:
            self.score_pulse_timer -= 1
            scale = 1.0 + (self.score_pulse_timer / Config.SCORE_PULSE_DURATION) * 0.15
            score_font = pygame.font.Font(None, int(48 * scale))
        else:
            score_font = self.font_large

        score_text = score_font.render(f"Score: {self.score}", True, Colors.DARK_GRAY)
        self.screen.blit(score_text, (10, 10))

        # High score
        high_score_text = self.font_small.render(f"High Score: {self.score_manager.high_score}", 
                                                True, Colors.GOLD)
        self.screen.blit(high_score_text, (10, 60))

        # Next value indicator
        next_text = self.font_small.render(f"Next: {self.next_value}", True, Colors.DARK_GRAY)
        self.screen.blit(next_text, (10, Config.SCREEN_HEIGHT - 30))

        # FPS counter
        fps_text = self.font_small.render(f"FPS: {int(self.clock.get_fps())}", True, Colors.DARK_GRAY)
        fps_rect = fps_text.get_rect(topright=(Config.SCREEN_WIDTH - 10, 10))
        self.screen.blit(fps_text, fps_rect)

    def render_game_over_screen(self) -> None:
        """Render game over overlay and interface."""
        if not self.game_over:
            return

        # Semi-transparent overlay
        overlay = pygame.Surface((Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 120))
        self.screen.blit(overlay, (0, 0))

        # Game over text
        game_over_text = self.font_large.render("GAME OVER", True, Colors.WHITE)
        game_over_rect = game_over_text.get_rect(center=(Config.SCREEN_WIDTH // 2, 
                                                        Config.SCREEN_HEIGHT // 2 - 50))
        self.screen.blit(game_over_text, game_over_rect)

        # Final score
        final_score_text = self.font_medium.render(f"Final Score: {self.score}", True, Colors.WHITE)
        final_score_rect = final_score_text.get_rect(center=(Config.SCREEN_WIDTH // 2, 
                                                           Config.SCREEN_HEIGHT // 2))
        self.screen.blit(final_score_text, final_score_rect)

        # New high score notification
        if self.new_high_score_achieved:
            new_high_text = self.font_medium.render("NEW HIGH SCORE!", True, Colors.GOLD)
            new_high_rect = new_high_text.get_rect(center=(Config.SCREEN_WIDTH // 2, 
                                                          Config.SCREEN_HEIGHT // 2 + 40))
            self.screen.blit(new_high_text, new_high_rect)

        # Game over buttons
        self.game_over_restart_btn.render(self.screen)
        self.game_over_exit_btn.render(self.screen)

        # Keyboard instructions
        keyboard_text = self.font_small.render("Press R to restart, ESC to quit", True, Colors.LIGHT_GRAY)
        keyboard_rect = keyboard_text.get_rect(center=(Config.SCREEN_WIDTH // 2, 
                                                      Config.SCREEN_HEIGHT // 2 + 145))
        self.screen.blit(keyboard_text, keyboard_rect)

    def render_frame(self) -> None:
        """Execute complete frame rendering pipeline."""
        self.render_background()
        self.render_game_area()
        self.render_preview_circle()

        # Render game objects
        for circle in self.circles:
            circle.render(self.screen)

        for particle in self.particles:
            particle.render(self.screen)

        # Render UI
        self.render_ui()

        if self.game_over:
            self.render_game_over_screen()
        else:
            self.restart_button.render(self.screen)
            self.exit_button.render(self.screen)

        pygame.display.flip()

    def handle_button_events(self, event: pygame.event.Event) -> Optional[str]:
        """
        Handle button interaction events.

        Args:
            event: Pygame event to process

        Returns:
            'exit' for quit action, 'restart' for restart, None otherwise
        """
        if self.restart_button.handle_event(event):
            self.restart_game()
            return 'restart'

        if self.exit_button.handle_event(event):
            return 'exit'

        if self.game_over:
            if self.game_over_restart_btn.handle_event(event):
                self.restart_game()
                return 'restart'
            if self.game_over_exit_btn.handle_event(event):
                return 'exit'

        return None

    def restart_game(self) -> None:
        """Reset game state for new session."""
        self.circles.clear()
        self.particles.clear()
        self.score = 0
        self.game_over = False
        self.new_high_score_achieved = False
        self.drop_x = Config.SCREEN_WIDTH // 2
        self.next_value = CircleManager.generate_next_value()
        self.next_radius = CircleManager.get_radius_for_value(self.next_value)
        self.score_pulse_timer = 0

    def handle_events(self) -> bool:
        """
        Process all pygame events.

        Returns:
            False if application should quit, True otherwise
        """
        for event in pygame.event.get():
            # Ignore window-close (pygame.QUIT) so program doesn't exit automatically.
            # If you prefer, you can set self.game_over = True instead of ignoring.
            if event.type == pygame.QUIT:
                # simply ignore close-button to force user to use the Exit button
                continue

            # Handle button events
            button_action = self.handle_button_events(event)
            if button_action == 'exit':
                return False
            elif button_action == 'restart':
                continue

            # Handle mouse clicks for dropping circles
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if not self.game_over:
                    mouse_x, mouse_y = event.pos
                    if (mouse_y > 55 and 
                        Config.PLAY_AREA_LEFT <= mouse_x <= Config.PLAY_AREA_RIGHT):
                        self.drop_circle()

            # Handle keyboard input
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and not self.game_over:
                    self.drop_circle()
                elif event.key == pygame.K_r and self.game_over:
                    self.restart_game()

                elif event.key == pygame.K_ESCAPE:
                    return False

        return True



    def run(self) -> None:
        """Execute main game loop."""
        running = True

        while running:
            running = self.handle_events()

            self.update_drop_position()

            if not self.game_over:
                self.update_physics()
                self.check_game_over_conditions()

            self.render_frame()
            self.clock.tick(Config.FPS)
            
        pygame.quit()
        sys.exit()

def get_base_dir() -> str:
    """
    Return directory next to the script/executable:
      - if frozen (exe) -> directory of the exe file (sys.argv[0])
      - else -> directory of the .py file
    """
    if getattr(sys, "frozen", False):
        # when packaged as exe, sys.argv[0] is the original exe path
        return os.path.dirname(os.path.abspath(sys.argv[0]))
    return os.path.dirname(os.path.abspath(__file__))



def main() -> None:
    """Application entry point."""
    try:
        game = GameEngine()
        game.run()
    except Exception as e:
        print(f"Game initialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
