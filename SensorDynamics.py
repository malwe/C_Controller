import numpy as np


# Sensor <-> Umgebung physikalisch PT1
# In Reihe dazu: EMA-Filter = PT1 (in etwa)
# => PT2 gesamt


class SensorDynamics:
    def __init__(self, dt, tau_T_sensor, tau_RH_sensor, tau_T_EMA, tau_RH_EMA, sigma_T, sigma_RH):
        self.dt = dt

        # PT1 - Sensor physikalische Dynamik
        self.tau_T_sensor = tau_T_sensor
        self.tau_RH_sensor = tau_RH_sensor

        self.sigma_T  = sigma_T
        self.sigma_RH = sigma_RH

        self.T_sensor = None
        self.RH_sensor = None

        # PT1 - Sensor algorithmische Dynamik durch EMA-Filter
        self.tau_T_EMA = tau_T_EMA
        self.tau_RH_EMA = tau_RH_EMA

        self.T_meas = None
        self.RH_meas = None

        # Es muss gelten für stabile Euler-Integration: DT / tau < 0.3
        assert self.dt / self.tau_T_sensor  < 0.3
        assert self.dt / self.tau_RH_sensor < 0.3
        assert self.dt / self.tau_T_EMA     < 0.3
        assert self.dt / self.tau_RH_EMA    < 0.3


    def init(self, T0, RH0):
        self.T_sensor = T0
        self.RH_sensor = RH0
        self.T_meas = T0
        self.RH_meas = RH0


    def step(self, T_true, RH_true):
        # PT1 zeitdiskret Euler-Vorwärts - Sensor physikalische Dynamik
        self.T_sensor += (self.dt / self.tau_T_sensor) * (T_true - self.T_sensor)
        self.RH_sensor += (self.dt / self.tau_RH_sensor) * (RH_true - self.RH_sensor)


        # Weißes Gauß-Rauschen (AWGN) erzeugen
        noise_T = np.random.normal(0.0, self.sigma_T)
        noise_RH = np.random.normal(0.0, self.sigma_RH)
        T_sensor_noisy  = self.T_sensor  + noise_T
        RH_sensor_noisy = self.RH_sensor + noise_RH


        # PT1 zeitdiskret Euler-Vorwärts - Sensor EMA-Filter Dynamik
        self.T_meas += (self.dt / self.tau_T_EMA) * (T_sensor_noisy - self.T_meas)
        self.RH_meas += (self.dt / self.tau_RH_EMA) * (RH_sensor_noisy - self.RH_meas)

        return self.T_meas, self.RH_meas