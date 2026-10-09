import matplotlib.pyplot as plt
import numpy as np


class Dithered_Delta_Sigma_PWM:
    def __init__(self, min_hold_time, dither_time, seed=0):
        self.min_hold_time = min_hold_time  # (s)
        self.dither_time = dither_time  # maximum additional hold time (s)
        if self.min_hold_time < 0.0 or self.dither_time < 0.0:
            raise ValueError("Hold times must be non-negative")

        self.rng = np.random.default_rng(seed)
        self.integrator = 0.0
        self.next_switch_time = -np.inf
        self.last_integration_time = None
        self.state = 0.0

    def step(self, u_continuous, simulation_time):
        assert u_continuous >= 0.0 and u_continuous <= 1.0

        if self.last_integration_time == None:
            self.last_integration_time = simulation_time

        self.integrator += (u_continuous - self.state) * (simulation_time - self.last_integration_time)
        self.integrator = np.clip(self.integrator, -5, 5)  # Anti-Windup
        self.last_integration_time = simulation_time

        if simulation_time >= self.next_switch_time:
            new_state = 0.0 if self.integrator < 0.0 else 1.0

            if new_state != self.state:
                self.state = new_state
                extra_hold_time = self.rng.uniform(0.0, self.dither_time)
                self.next_switch_time = simulation_time + self.min_hold_time + extra_hold_time

        return self.state


pwm_timestep = 0.01023
motorcontroller_timestep = 1.013

u = 0.45
simulation_duration = 600.0
min_hold_time = 0.4
# max_dither_time = max(min_hold_time, pwm_timestep*2)
max_dither_time = 0

pwm_time = np.arange(int(simulation_duration / pwm_timestep) + 1) * pwm_timestep
motorcontroller_time = np.arange(int(simulation_duration / motorcontroller_timestep) + 1) * motorcontroller_timestep

pwm = Dithered_Delta_Sigma_PWM(min_hold_time, max_dither_time, seed=0)


pwm_output = np.empty(pwm_time.size)

for i in range(pwm_time.size):
    pwm_output[i] = pwm.step(u, pwm_time[i])


on_cnt = 1
off_cnt = 1
last_state = 0
current_duty_cycle = 0.0
pwm_duty_cycle = np.empty(pwm_time.size)

for i in range(pwm_time.size):

    if last_state == 0 and pwm_output[i] == 1:
        last_state = 1
        current_duty_cycle = on_cnt / (on_cnt + off_cnt)
        on_cnt = 0
    elif last_state == 1 and pwm_output[i] == 0:
        last_state = 0
        current_duty_cycle = on_cnt / (on_cnt + off_cnt)
        off_cnt = 0

    if pwm_output[i] == 0:
        off_cnt += 1
    else:
        on_cnt += 1

    pwm_duty_cycle[i] = current_duty_cycle


previous_pwm_index = np.searchsorted(pwm_time, motorcontroller_time, side="right") - 1
motorcontroller_input = pwm_output[previous_pwm_index]


on_cnt = 1
off_cnt = 1
last_state = 0
current_duty_cycle = 0.0
motorcontroller_duty_cycle = np.empty(motorcontroller_time.size)

for i in range(motorcontroller_time.size):

    if last_state == 0 and motorcontroller_input[i] == 1:
        last_state = 1
        current_duty_cycle = on_cnt / (on_cnt + off_cnt)
        on_cnt = 0
    elif last_state == 1 and motorcontroller_input[i] == 0:
        last_state = 0
    #     current_duty_cycle = on_cnt / (on_cnt + off_cnt)
        off_cnt = 0

    if motorcontroller_input[i] == 0:
        off_cnt += 1
    else:
        on_cnt += 1

    motorcontroller_duty_cycle[i] = current_duty_cycle


motorcontroller_filter_tau = 5.0  # Low-pass time constant (s)

motorcontroller_duty_cycle_filtered = np.empty(motorcontroller_time.size)
motorcontroller_duty_cycle_filtered[0] = motorcontroller_duty_cycle[0]

for i in range(1, motorcontroller_time.size):

    alpha = 1.0 - np.exp(-motorcontroller_timestep / motorcontroller_filter_tau)

    motorcontroller_duty_cycle_filtered[i] = (
        motorcontroller_duty_cycle_filtered[i - 1]
        + alpha * (motorcontroller_duty_cycle[i] - motorcontroller_duty_cycle_filtered[i - 1])
    )


fig, axes = plt.subplots(5, 1, sharex=True, figsize=(10, 8))

axes[0].step(pwm_time, pwm_output, where="post")
axes[0].set_ylabel("PWM-Ausgabe")
axes[0].set_ylim(-0.1, 1.1)
axes[0].grid(True)

axes[1].step(pwm_time, pwm_duty_cycle, where="post")
axes[1].set_ylabel("PWM-Duty-Cycle")
axes[1].set_ylim(-0.1, 1.1)
axes[1].grid(True)

axes[2].step(motorcontroller_time, motorcontroller_input, where="post")
axes[2].set_ylabel("Motorcontroller-Eingabe")
axes[2].set_ylim(-0.1, 1.1)
axes[2].grid(True)

axes[3].step(motorcontroller_time, motorcontroller_duty_cycle, where="post")
axes[3].set_ylabel("Motorcontroller-Duty-Cycle")
axes[3].set_ylim(-0.1, 1.1)
axes[3].grid(True)

axes[4].plot(motorcontroller_time, motorcontroller_duty_cycle_filtered)
axes[4].set_ylabel("Gefilterter\nDuty-Cycle")
axes[4].set_ylim(-0.1, 1.1)
axes[4].grid(True)
axes[4].set_xlabel("Zeit (s)")

fig.suptitle(f"Delta-Sigma-Ausgaben bei u = {u:.2f}")
fig.tight_layout()
plt.show()
