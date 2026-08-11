import carla
import weakref
import math


class CollisionSensor:
    """Collision sensor aligned with ScenarioRunner's CollisionTest filters.

    Filters:
    - Ego speed < 0.1 m/s → not ego's fault
    - Unknown actor classes and sidewalks are excluded
    - Duplicate contacts with the same actor/time/location are collapsed
    """

    COLLISION_RADIUS_M = 5.0
    DUPLICATE_WINDOW_FRAMES = 100  # 5 seconds at the 20 Hz benchmark rate

    def __init__(self, parent_actor):
        self.sensor = None
        self.history = []
        self.parent = parent_actor
        self._last_actor_id = None
        self._last_frame = None
        self._last_location = None

        world = parent_actor.get_world()
        blueprint_library = world.get_blueprint_library()

        bp = blueprint_library.find('sensor.other.collision')

        self.sensor = world.spawn_actor(
            bp,
            carla.Transform(),
            attach_to=parent_actor
        )

        weak_self = weakref.ref(self)

        self.sensor.listen(
            lambda event: CollisionSensor._on_collision(
                weak_self,
                event
            )
        )

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()

        if not self:
            return

        # Speed filter: stationary ego (speed < 0.1 m/s) → not ego's fault.
        # Matches Bench2Drive evaluation criterion. Note: this excludes legitimate
        # low-speed collisions, biasing the dataset toward higher-speed events.
        vel = self.parent.get_velocity()
        ego_speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        if ego_speed < 0.1:
            return

        actor = event.other_actor
        actor_id = actor.id if actor is not None else -1
        actor_type = actor.type_id if actor is not None else 'world'
        if not (
            (('static' in actor_type or 'traffic' in actor_type) and 'sidewalk' not in actor_type)
            or 'vehicle' in actor_type
            or 'walker' in actor_type
        ):
            return

        location = self.parent.get_location()
        if (
            self._last_actor_id == actor_id
            and self._last_frame is not None
            and event.frame - self._last_frame <= self.DUPLICATE_WINDOW_FRAMES
        ):
            return
        if self._last_location is not None:
            dx = location.x - self._last_location[0]
            dy = location.y - self._last_location[1]
            dz = location.z - self._last_location[2]
            if math.sqrt(dx * dx + dy * dy + dz * dz) <= self.COLLISION_RADIUS_M:
                return

        collision_info = {
            'frame': event.frame,
            'actor_id': actor_id,
            'actor_type': actor_type,
            'impulse': (
                event.normal_impulse.x,
                event.normal_impulse.y,
                event.normal_impulse.z
            )
        }

        self.history.append(collision_info)
        self._last_actor_id = actor_id if actor_id != 0 else None
        self._last_frame = int(event.frame)
        self._last_location = (location.x, location.y, location.z)

        print(f"[COLLISION] with {actor_type}")

    def has_collision(self):
        return len(self.history) > 0

    def get_latest_collision(self):
        if len(self.history) == 0:
            return None

        return self.history[-1]

    def clear(self):
        self.history.clear()
        self._last_actor_id = None
        self._last_frame = None
        self._last_location = None

    def destroy(self):
        """Silently clean up sensor - CARLA may have already destroyed it."""
        if self.sensor is not None:
            try:
                self.sensor.stop()
                self.sensor.destroy()
            except Exception:
                pass  # Silently ignore all errors
            finally:
                self.sensor = None
        self.history.clear()
        self._last_actor_id = None
        self._last_frame = None
        self._last_location = None
