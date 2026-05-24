# Sensor <-> Umgebung physikalisch PT1 und EMA-Filter PT1    => PT2 gesamt


class SensorDynamics:
    def __init__(self, dt, tau_T_sensor, tau_RH_sensor, tau_T_EMA, tau_RH_EMA):
        self.dt = dt

        # PT1 - Sensor physikalische Dynamik
        self.tau_T = tau_T_sensor
        self.tau_RH = tau_RH_sensor

        # PT1 - Sensor algorithmische Dynamik durch EMA-Filter
        self.tau_T_EMA = tau_T_EMA
        self.tau_RH_EMA = tau_RH_EMA

        self.T_meas = None
        self.RH_meas = None

    def init(self, T0, RH0):
        self.T_meas = T0
        self.RH_meas = RH0

    def step(self, T_true, RH_true):
        # PT1 zeitdiskret Euler-Vorwärts - Sensor physikalische Dynamik
        self.T_meas += (self.dt / self.tau_T) * (T_true - self.T_meas)
        self.RH_meas += (self.dt / self.tau_RH) * (RH_true - self.RH_meas)

        # PT1 zeitdiskret Euler-Vorwärts - Sensor EMA-Filter Dynamik
        self.T_meas += (self.dt / self.tau_T_EMA) * (T_true - self.T_meas)
        self.RH_meas += (self.dt / self.tau_RH_EMA) * (RH_true - self.RH_meas)

        return self.T_meas, self.RH_meas