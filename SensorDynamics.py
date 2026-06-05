import numpy as np


# Sensor <-> Umgebung physikalisch PT1
# In Reihe dazu: EMA-Filter = PT1
# => PT2 gesamt
# G(s) = 1 / (1 + tau_sensor) * 1 / (1 + tau_EMA)


class SensorDynamics:
    def __init__(self, dt, tau_T_EMA, tau_RH_EMA, sigma_T, sigma_RH):
        self.dt = dt

        self.sigma_T  = sigma_T
        self.sigma_RH = sigma_RH

        # PT1 - Sensor algorithmische Dynamik durch EMA-Filter
        self.tau_T_EMA = tau_T_EMA
        self.tau_RH_EMA = tau_RH_EMA

        self.T_meas = None
        self.RH_meas = None

        # Es muss gelten für stabile Euler-Integration: DT / tau < 0.3
        assert tau_T_EMA  == 0 or self.dt / self.tau_T_EMA  < 0.3
        assert tau_RH_EMA == 0 or self.dt / self.tau_RH_EMA < 0.3


    def init(self, T0, RH0):
        self.T_meas = T0
        self.RH_meas = RH0


    def step(self, T_sensor, RH_sensor):
        # Weißes Gauß-Rauschen (AWGN) erzeugen
        noise_T = np.random.normal(0.0, self.sigma_T)
        noise_RH = np.random.normal(0.0, self.sigma_RH)
        T_sensor_noisy  = T_sensor  + noise_T
        RH_sensor_noisy = RH_sensor + noise_RH

        # PT1/ EMA-Filter
        if self.tau_T_EMA > 0:
            self.T_meas += (self.dt / self.tau_T_EMA) * (T_sensor_noisy - self.T_meas)
        else:
            self.T_meas = T_sensor_noisy

        if self.tau_RH_EMA > 0:
            self.RH_meas += (self.dt / self.tau_RH_EMA) * (RH_sensor_noisy - self.RH_meas)
        else:
            self.RH_meas = RH_sensor_noisy

        return self.T_meas, self.RH_meas