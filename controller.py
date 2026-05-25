from shared import *


class Controller:
    def __init__(self, dt):
        self.dt = dt

        self.last_control_time = 240.0 * 60.0  # (s)

        self.u_f = 0.0
        self.u_h = 0.0


    def compute_control(self, simulation_time, T_meas, VPD_atm_meas):

        if simulation_time - self.last_control_time >= 5.0:  # (s) Regelintervall
            self.last_control_time = simulation_time
            
            if VPD_atm_meas < VPD_MIN:    # to humid
                self.u_h = 0.0
                self.u_f = U_F_ON
            elif VPD_atm_meas > VPD_MAX:  # to dry
                self.u_h = U_H_ON
                self.u_f = 0.0
            else:
                self.u_h = 0.0
                self.u_f = 0.0

            if T_meas > T_MAX:            # to hot
                self.u_f = U_F_ON

        return self.u_f, self.u_h