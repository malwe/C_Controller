class Controller:
    def __init__(self, dt):
        self.dt = dt

        self.last_control_time = 240.0 * 60.0  # (s)

        self.U_F_ON = 0.5  # max Lüfterleistung (0-1)
        self.U_H_ON = 1.0  # max Humidifier-Leistung (0-1)

        self.VPD_MIN = 1.2  # (kPa)
        self.VPD_MAX = 1.3  # (kPa)
        self.T_MAX = 27.5   # (°C)

        self.u_f = 0.0
        self.u_h = 0.0


    def compute_control(self, simulation_time, T_meas, VPD_atm_meas):

        if simulation_time - self.last_control_time >= 5.0:  # (s) Regelintervall
            self.last_control_time = simulation_time
            
            if VPD_atm_meas < self.VPD_MIN:    # to humid
                self.u_h = 0.0
                self.u_f = self.U_F_ON
            elif VPD_atm_meas > self.VPD_MAX:  # to dry
                self.u_h = self.U_H_ON
                self.u_f = 0.0
            else:
                self.u_h = 0.0
                self.u_f = 0.0

            if T_meas > self.T_MAX:            # to hot
                self.u_f = self.U_F_ON

        return self.u_f, self.u_h