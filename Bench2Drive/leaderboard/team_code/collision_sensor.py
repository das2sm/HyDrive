import carla
import weakref


class CollisionSensor:
    def __init__(self, parent_actor):
        self.sensor = None
        self.history = []

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

        actor = event.other_actor

        collision_info = {
            'frame': event.frame,
            'actor_id': actor.id,
            'actor_type': actor.type_id,
            'impulse': (
                event.normal_impulse.x,
                event.normal_impulse.y,
                event.normal_impulse.z
            )
        }

        self.history.append(collision_info)

        print(f"[COLLISION] with {actor.type_id}")

    def has_collision(self):
        return len(self.history) > 0

    def get_latest_collision(self):
        if len(self.history) == 0:
            return None

        return self.history[-1]

    def clear(self):
        self.history.clear()

    def destroy(self):
        """Silently clean up sensor - CARLA may have already destroyed it."""
        if self.sensor is not None:
            try:
                self.sensor.stop()
                self.sensor.destroy()
            except:
                pass  # Silently ignore all errors
            finally:
                self.sensor = None
        self.history.clear()