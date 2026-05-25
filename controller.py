from shared import *
import shared
import numpy as np


class Controller:
    def __init__(self, dt):
        self.dt = dt

        self.last_control_time = 0.0  # (s)

        self.control_time_step = 1.0  # (s)

        self.u_f = 0.0
        self.u_h = 0.0


        # Piecewise-linear gain scheduling / continuous heuristic nonlinear controller / piecewise linear fuzzy-like controller
        self.plgs_vpd_transition_width = (VPD_MAX - VPD_MIN) / 2  # Keine Totzone -> quasi P-Regler mit Stellgrenzen
        self.plgs_vpd_min    = VPD_MIN  # - self.plgs_vpd_transition_width / 2
        self.plgs_vpd_good_1 = self.plgs_vpd_min + self.plgs_vpd_transition_width
        self.plgs_vpd_max    = VPD_MAX  # + self.plgs_vpd_transition_width / 2
        self.plgs_vpd_good_2 = self.plgs_vpd_max - self.plgs_vpd_transition_width
        #
        self.plgs_T_transition_width = T_MAX - T_GOOD
        self.plgs_T_max = T_MAX
        self.plgs_T_good_2 = self.plgs_T_max - self.plgs_T_transition_width


        # PI_T_and_PI_VPD
        self.T_set = (T_MAX + T_GOOD) / 2  # REVIEW: über diese Definition lässt sich streiten
        self.vpd_set = (VPD_MIN + VPD_MAX) / 2

        # PI gains
        self.Kp_f_T = 0.4
        self.Ki_f_T = 0.001

        self.Kp_f_vpd = 0.6
        self.Ki_f_vpd = 0.002

        self.Kp_h_vpd = 7.0
        self.Ki_h_vpd = 0.005

        # Integrator states
        self.I_f_T = 0.0
        self.I_f_vpd = 0.0
        self.I_h_vpd = 0.0

        # Anti-windup limits
        self.I_limit_f = 1.0
        self.I_limit_h = 1.0

    def bang_bang(self, T, VPD):

        if shared.simulation_time - self.last_control_time >= self.control_time_step:
            self.last_control_time = shared.simulation_time
            
            # low priority = humidity
            if VPD < VPD_MIN:               # too humid
                self.u_h = 0.0
                self.u_f = U_F_ON
            elif VPD > VPD_MAX:             # too dry
                self.u_h = U_H_ON
                self.u_f = 0.0
            else:                           # humidity good
                self.u_h = 0.0
                self.u_f = 0.0

            # high priority = temperature
            if T > T_MAX:                   # too hot
                self.u_f = U_F_ON

        assert(self.u_f >= 0 and self.u_f <= 1 and self.u_h >= 0 and self.u_h <= 1)
        return self.u_f, self.u_h
    

    # piecewise-linear gain scheduling - heuristisch - Ähnlichkeiten mit P-Regler
    def plgs(self, T, VPD):

        if shared.simulation_time - self.last_control_time >= self.control_time_step:
            self.last_control_time = shared.simulation_time

            u_h_VPD = 0.0
            u_f_VPD = 0.0
            u_f_T = 0.0

            if VPD < self.plgs_vpd_min:            # too humid
                u_h_VPD = 0.0
                u_f_VPD = U_F_ON
            elif VPD < self.plgs_vpd_good_1:       # humid but ok
                u_h_VPD = 0.0
                u_f_VPD = U_F_ON * (1 - (VPD - self.plgs_vpd_min) / self.plgs_vpd_transition_width)
            elif VPD < self.plgs_vpd_good_2:       # humidity perfect
                u_h_VPD = 0.0
                u_f_VPD = 0.0
            elif VPD < self.plgs_vpd_max:          # dry but ok
                u_h_VPD = U_H_ON * (VPD - self.plgs_vpd_good_2) / self.plgs_vpd_transition_width
                u_f_VPD = 0.0
            else:
                u_h_VPD = U_H_ON
                u_f_VPD = 0.0

            if (T > self.plgs_T_good_2) and (T < self.plgs_T_max):  # hot but ok
                u_f_T = U_F_ON * (T - self.plgs_T_good_2) / self.plgs_T_transition_width
            elif T >= self.plgs_T_max:              # too hot
                u_f_T = U_F_ON

            self.u_h = u_h_VPD
            self.u_f = max(u_f_VPD, u_f_T)

        assert(self.u_f >= 0 and self.u_f <= 1 and self.u_h >= 0 and self.u_h <= 1)
        return self.u_f, self.u_h
    

    def PI_T_and_PI_VPD(self, T, VPD):

        # Einseitige Leaky Integratoren / Integrator Decay
        # Conditional Integration - Anti-Windup

        if shared.simulation_time - self.last_control_time >= self.control_time_step:
            self.last_control_time = shared.simulation_time

            e_T = max(0.0, T - self.T_set)
            self.I_f_T += self.Ki_f_T * e_T * self.control_time_step
            self.I_f_T = np.clip(self.I_f_T, 0.0, self.I_limit_f)
            u_f_T = self.Kp_f_T * e_T + self.I_f_T

            e_vpd = VPD - self.vpd_set
            e_dry = max(0.0, e_vpd)  # Zu trocken -> humidifier
            self.I_h_vpd += self.Ki_h_vpd * e_dry * self.control_time_step
            self.I_h_vpd = np.clip(self.I_h_vpd, 0.0, self.I_limit_h)
            u_h_vpd = self.Kp_h_vpd * e_dry + self.I_h_vpd

            e_humid = max(0.0, -e_vpd)  # Zu feucht -> Lüfter
            self.I_f_vpd += self.Ki_f_vpd * e_humid * self.control_time_step
            self.I_f_vpd = np.clip(self.I_f_vpd, 0.0, self.I_limit_f)
            u_f_vpd = self.Kp_f_vpd * e_humid + self.I_f_vpd

            self.u_f = max(u_f_T, u_f_vpd)
            self.u_h = u_h_vpd

            self.u_f = np.clip(self.u_f, 0.0, 1.0)
            self.u_h = np.clip(self.u_h, 0.0, 1.0)

        assert(self.u_f >= 0 and self.u_f <= 1 and self.u_h >= 0 and self.u_h <= 1)
        return self.u_f, self.u_h, self.I_f_T, self.I_f_vpd, self.I_h_vpd


    def PI_T_and_PI_w(self, T, VPD):
        return 0