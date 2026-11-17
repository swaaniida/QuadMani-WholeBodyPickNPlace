"""PiPER command-side limits; no claim of an identified motor/firmware model."""
import torch


class ArmCommandFilter:
    """Bound transmitted joint-target slew and optionally model measured delay.

    Velocity limits bound command slew, not guaranteed measured joint velocity.
    No guessed acceleration, continuous torque, or CAN latency is inserted.
    """
    def __init__(self, initial, lower, upper, velocity, dt, delay_steps=0):
        if dt <= 0 or int(delay_steps) != delay_steps or delay_steps < 0:
            raise ValueError('Invalid arm command dt/delay')
        self.lower, self.upper = lower, upper
        self.max_delta = torch.as_tensor(velocity, device=initial.device, dtype=initial.dtype) * dt
        if torch.any(self.max_delta <= 0):
            raise ValueError('Arm command velocity must be positive')
        self.delay_steps = int(delay_steps)
        self.last = initial.clone()
        self.history = initial[:, None, :].repeat(1, self.delay_steps + 1, 1)

    def reset(self, env_ids, position):
        self.last[env_ids] = position
        self.history[env_ids] = position[:, None, :]

    def step(self, desired):
        bounded = torch.maximum(torch.minimum(desired, self.upper), self.lower)
        self.history = torch.cat([self.history[:, 1:], bounded[:, None, :]], dim=1)
        delayed = self.history[:, 0]
        delta = torch.maximum(torch.minimum(delayed - self.last, self.max_delta), -self.max_delta)
        self.last = torch.maximum(torch.minimum(self.last + delta, self.upper), self.lower)
        return self.last.clone()


def arm_drive_properties(properties, strength, stiffness, damping, velocity):
    """Scale effective position-drive response, leaving leg/finger properties alone.

    Gain scaling approximates response variation. Effort saturation is retained;
    this is not an identified torque-constant/current-limit motor model.
    """
    result = properties.copy()
    result['stiffness'][12:18] = stiffness * strength
    result['damping'][12:18] = damping * strength
    result['velocity'][12:18] = velocity
    return result
