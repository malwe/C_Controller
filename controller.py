from shared import *
import shared
import numpy as np


CONTROL_DT = 1.0

count = 0


class PID_Controller:
    def __init__(self, Kp, Ki, Kd, I_limit, I_leak, tau_D):

        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.I_limit = I_limit
        self.I_leak = I_leak
        self.tau_D = tau_D

        self.last_control_time = 0.0
        self.last_filter_time = 0.0

        self.I = 0.0
        self.e_D = 0.0
        self.e_D_last = 0.0


    def get_control_variable(self, e):
        # Leaky Integrator
        self.I *= max(0.0, (1.0 - self.I_leak * (shared.simulation_time - self.last_control_time)))

        # PID Rohsignal mit altem I für conditional integration condition
        u_condition = self._calculate_u(e)

        # Conditional Integration
        # I-Anteil soll nur kleine statische Regelabweichung des P-Reglers ausgleichen
        # -> keine Einstellung bei starken Regelabweichungen (= während starker Dynamik)
        # REVIEW: Evtl. würde es sich lohnen die Schwelle runterzusetzen
        if not (u_condition >= 1.0):
            self.I += self.Ki * e * (shared.simulation_time - self.last_control_time)

        self.I = np.clip(self.I, -self.I_limit, self.I_limit)

        # PID Rohsignal mit aktuellem I
        u = self._calculate_u(e)

        self.e_D_last = self.e_D
        self.last_control_time = shared.simulation_time
        return u

    
    # Bei jedem neuen Messwert aufrufen
    def update_e_D(self, e):
        # PT1 Tiefpass (auch als EMA umsetzbar) für e_D
        dt_filter = shared.simulation_time - self.last_filter_time
        assert dt_filter / self.tau_D < 0.3, "PT1 e_D-Filter numerisch instabil"
        self.e_D += (dt_filter / self.tau_D) * (e - self.e_D)
        self.last_filter_time = shared.simulation_time


    def _calculate_u(self, e):
        dt_control = shared.simulation_time - self.last_control_time
        _pi = self.Kp * e + self.I
        if dt_control <= 0:
            return _pi
        else:
            return _pi + self.Kd * (self.e_D - self.e_D_last) / (shared.simulation_time - self.last_control_time)
    

    def get_e_D(self):
        return self.e_D
    

class T_and_VPD_PID_Controller:
    def __init__(self, dt):

        self.last_control_time = 0.0  # (s)

        # Möglichst bei jedem Messwert PID berechnen, da Delta_Sigma_PWM sonst zeitlich stark quantisiert wird
        self.control_dt = 0.0  # (s) kann beliebig schnell sein, da Delta_Sigma_PWM keine feste PWM-Periode braucht

        self.u_f = 0.0
        self.u_h = 0.0

        self.T_set = 25.0
        self.vpd_set = 1.25

        # EINSTELLEN
        # Regelgrenzen -> Bestimmen Kp!
        T_set = 25  # u_f_T = 0
        T_max = 29  # u_f_T = U_F_ON (in Realität 100%)

        vpd_min = 1.05
        vpd_set = 1.25
        vpd_max = 1.35

        # PID gains
        Kp_f_T = 1 / (T_max - T_set)
        Ki_f_T = 0.0
        Kd_f_T = 0.7 * np.sqrt(Kp_f_T)  # Faktor typisch im Bereich 0.3...1  # EINSTELLEN

        Kp_f_vpd = 1 / (vpd_set - vpd_min)
        Ki_f_vpd = 0.0
        Kd_f_vpd = 0.7 * np.sqrt(Kp_f_vpd)  # Faktor typisch im Bereich 0.3...1  # EINSTELLEN

        Kp_h_vpd = 1 / (vpd_max - vpd_set)
        Ki_h_vpd = 0.0
        Kd_h_vpd = 0.7 * np.sqrt(Kp_h_vpd)  # Faktor typisch im Bereich 0.3...1  # EINSTELLEN

        # EINSTELLEN
        # Anti-windup limits - I in [0, limit] erlaubt
        I_limit_f_T   = 0.0
        I_limit_f_vpd = 0.0
        I_limit_h_vpd = 0.0

        # EINSTELLEN
        # Integrator leakage per second
        # I *= (1.0 - leak * control_dt)
        # Bedingung: 0 <= leak * control_dt < 1
        I_leak_f_T   = 0.005
        I_leak_f_vpd = 0.005
        I_leak_h_vpd = 0.005

        # EINSTELLEN
        # PT1 / EMA - Tiefpassfilter für D-Anteil - zusätzlich zu allgemeinem Filter auf Messwerte
        tau_D_f_T   = 3.0    # (s)
        tau_D_f_vpd = 3.0
        tau_D_h_vpd = 3.0

        self.pid_f_T =   PID_Controller(Kp_f_T,   Ki_f_T,   Kd_f_T,   I_limit_f_T,   I_leak_f_T,   tau_D_f_T)
        self.pid_f_vpd = PID_Controller(Kp_f_vpd, Ki_f_vpd, Kd_f_vpd, I_limit_f_vpd, I_leak_f_vpd, tau_D_f_vpd)
        self.pid_h_vpd = PID_Controller(Kp_h_vpd, Ki_h_vpd, Kd_h_vpd, I_limit_h_vpd, I_leak_h_vpd, tau_D_h_vpd)

        # min_pwm_hold_time_fan = 7.0  # (s) Minimale Zeit PWM an/aus
        # min_pwm_hold_time_humidifier = 3.0
        # self.fan_quantizer = Delta_Sigma_PWM(min_pwm_hold_time_fan)
        # self.humidifier_quantizer = Delta_Sigma_PWM(min_pwm_hold_time_humidifier)

        min_quant_hold_time_fan = 1.0  # (s) Minimale Haltezeit des quantisierten Wertes
        min_quant_hold_time_humidifier = 1.0
        num_bits_quantizer = 3
        self.fan_quantizer = Delta_Sigma_Quantizer(min_quant_hold_time_fan, num_bits_quantizer)
        self.humidifier_quantizer = Delta_Sigma_Quantizer(min_quant_hold_time_humidifier, num_bits_quantizer)

    
    def get_control_variables(self, T, VPD, f_pwm):
        e_f_T_raw = T - self.T_set
        e_vpd_raw = VPD - self.vpd_set
        e_f_vpd_raw = -e_vpd_raw
        e_h_vpd_raw = e_vpd_raw

        self.pid_f_T.update_e_D(e_f_T_raw)
        self.pid_f_vpd.update_e_D(e_f_vpd_raw)
        self.pid_h_vpd.update_e_D(e_h_vpd_raw)

        if shared.simulation_time - self.last_control_time >= self.control_dt:
            self.last_control_time = shared.simulation_time

            e_f_T = max(0.0, T - self.T_set)  # zu warm
            e_vpd = VPD - self.vpd_set
            e_h_vpd = max(0.0, e_vpd)         # zu trocken
            e_f_vpd = max(0.0, -e_vpd)        # zu feucht

            u_f_T_raw = self.pid_f_T.get_control_variable(e_f_T)
            u_f_vpd_raw = self.pid_f_vpd.get_control_variable(e_f_vpd)
            u_h_vpd_raw = self.pid_h_vpd.get_control_variable(e_h_vpd)

            # Priorität bekommt der Regler mit größerem Stellwert
            #   -> (hartes) Umschalten des dominanten Reglers -> evtl. Integrator-Mismatch
            # REVIEW: Gewichtetes Mittel besser?
            self.u_f = max(u_f_T_raw, u_f_vpd_raw) 
            self.u_h = u_h_vpd_raw

            self.u_f = np.clip(self.u_f, 0.0, 1.0)
            self.u_h = np.clip(self.u_h, 0.0, 1.0)

            # nur für Simulation (Potentiometereinstellung an Abluft und Humidifier)
            self.u_f *= U_F_ON
            self.u_h *= U_H_ON

        assert(self.u_f >= 0 and self.u_f <= 1 and self.u_h >= 0 and self.u_h <= 1)

        if f_pwm:
            fan_pwm_state = self.fan_quantizer.step(self.u_f / U_F_ON)  # PWM: kontinuierlich[0, 1] -> diskret{0, 1}
            humidifier_pwm_state = self.humidifier_quantizer.step(self.u_h / U_H_ON)

            self.u_f = fan_pwm_state * U_F_ON
            self.u_h = humidifier_pwm_state * U_H_ON

        return self.u_f, self.u_h
    

    # Zum Plotten in main.py
    def get_D_filtered(self):
        D_f_T = self.T_set + self.pid_f_T.get_e_D()
        D_f_vpd = self.vpd_set - self.pid_f_vpd.get_e_D()
        D_h_vpd = self.vpd_set + self.pid_h_vpd.get_e_D()
        return D_f_T, D_f_vpd, D_h_vpd


class PLGS_Controller:
    def __init__(self):

        self.last_control_time = 0.0  # (s)

        self.control_dt = CONTROL_DT  # (s)

        self.u_f = 0.0
        self.u_h = 0.0

        # PWM on time
        # PWM period = self.control_dt
        self.on_time_fan = 0.0
        self.on_time_hum = 0.0

        # Piecewise-linear gain scheduling / continuous heuristic nonlinear controller / piecewise linear fuzzy-like controller
        # self.plgs_vpd_transition_width = (VPD_MAX - VPD_MIN) / 2  # Keine Totzone -> quasi P-Regler
        # self.plgs_vpd_min    = VPD_MIN  # - self.plgs_vpd_transition_width / 2
        # self.plgs_vpd_good_1 = self.plgs_vpd_min + self.plgs_vpd_transition_width
        # self.plgs_vpd_max    = VPD_MAX  # + self.plgs_vpd_transition_width / 2
        # self.plgs_vpd_good_2 = self.plgs_vpd_max - self.plgs_vpd_transition_width
        self.plgs_vpd_min = 1.1
        self.plgs_vpd_good_1 = 1.25
        self.plgs_vpd_good_2 = 1.25
        self.plgs_vpd_max = 1.4
        #
        # self.plgs_T_transition_width = T_MAX - T_GOOD
        # self.plgs_T_max = T_MAX
        # self.plgs_T_good_2 = self.plgs_T_max - self.plgs_T_transition_width
        self.plgs_T_good_2 = 25.0
        self.plgs_T_max = 28


        # piecewise-linear gain scheduling - heuristisch - Ähnlichkeiten mit P-Regler
    def get_control_variables(self, T, VPD, f_pwm):

        if shared.simulation_time - self.last_control_time >= self.control_dt:
            self.last_control_time = shared.simulation_time

            u_h_VPD = 0.0
            u_f_VPD = 0.0
            u_f_T = 0.0

            if VPD < self.plgs_vpd_min:            # too humid
                u_h_VPD = 0.0
                u_f_VPD = U_F_ON
            elif VPD < self.plgs_vpd_good_1:       # humid but ok
                u_h_VPD = 0.0
                u_f_VPD = U_F_ON * (1 - (VPD - self.plgs_vpd_min) / (self.plgs_vpd_good_1 - self.plgs_vpd_min))
            elif VPD < self.plgs_vpd_good_2:       # humidity perfect
                u_h_VPD = 0.0
                u_f_VPD = 0.0
            elif VPD < self.plgs_vpd_max:          # dry but ok
                u_h_VPD = U_H_ON * (VPD - self.plgs_vpd_good_2) / (self.plgs_vpd_max - self.plgs_vpd_good_2)
                u_f_VPD = 0.0
            else:
                u_h_VPD = U_H_ON
                u_f_VPD = 0.0

            if (T > self.plgs_T_good_2) and (T < self.plgs_T_max):  # hot but ok
                u_f_T = U_F_ON * (T - self.plgs_T_good_2) / (self.plgs_T_max - self.plgs_T_good_2)
            elif T >= self.plgs_T_max:              # too hot
                u_f_T = U_F_ON

            self.u_h = u_h_VPD
            self.u_f = max(u_f_VPD, u_f_T)  # Priorität bekommt der Regler mit größerem Stellwert

            # PWM
            self.on_time_fan = self.u_f / U_F_ON * self.control_dt
            self.on_time_hum = self.u_h / U_H_ON * self.control_dt

            if self.on_time_fan < 1.0: self.on_time_fan = 0.0
            if self.control_dt - self.on_time_fan < 1.0: self.on_time_fan = self.control_dt

            if self.on_time_hum < 1.0: self.on_time_hum = 0.0
            if self.control_dt - self.on_time_hum < 1.0: self.on_time_hum = self.control_dt

        u_f_pwm = U_F_ON if shared.simulation_time - self.last_control_time < self.on_time_fan else 0.0
        u_h_pwm = U_H_ON if shared.simulation_time - self.last_control_time < self.on_time_hum else 0.0

        # assert(self.u_f >= 0 and self.u_f <= 1 and self.u_h >= 0 and self.u_h <= 1)
        if f_pwm:
            return u_f_pwm, u_h_pwm
        else:
            return self.u_f, self.u_h


class Bang_Bang_Controller:
    def __init__(self, ):
        self.last_control_time = 0.0  # (s)

        self.control_dt = CONTROL_DT  # (s)

        self.u_f = 0.0
        self.u_h = 0.0


    def get_control_variables(self, T, VPD):

        if shared.simulation_time - self.last_control_time >= self.control_dt:
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
    

# first-order 1-bit delta-sigma modulator with dwell-time-limited switching and anti-windup
# Nicht ganz/ abgewandelter Spezialfall von Delta_Sigma_Quantizer siehe unten
class Delta_Sigma_PWM:
    def __init__(self, min_hold_time):
        self.min_hold_time = min_hold_time  # (s)
        self.integrator = 0.0
        self.last_switch_time = -1e9
        self.last_integration_time = None
        self.state = 0.0

    def step(self, u_continuous):
        assert u_continuous >= 0.0 and u_continuous <= 1.0

        if self.last_integration_time == None:
            self.last_integration_time = shared.simulation_time
        
        self.integrator += (u_continuous - self.state) * (shared.simulation_time - self.last_integration_time)
        self.integrator = np.clip(self.integrator, -5, 5)  # Anti-Windup
        self.last_integration_time = shared.simulation_time

        if (shared.simulation_time - self.last_switch_time) >= self.min_hold_time:
            new_state = 0.0 if self.integrator < 0.0 else 1.0

            if new_state != self.state:
                self.state = new_state
                self.last_switch_time = shared.simulation_time
        
        return self.state


# first-order N-bit delta-sigma modulator with dwell-time-limited switching
# rate-limited multi-level tracking quantizer
class Delta_Sigma_Quantizer:
    def __init__(self, min_hold_time, num_bits):
        assert num_bits >= 1

        self.num_bits = num_bits
        self.num_levels = 2**num_bits
        self.levels = np.linspace(0.0, 1.0, self.num_levels)
        self.min_hold_time = min_hold_time

        self.integrator = 0.0  # Fehlerzustand
        self.output = 0.0
        self.last_integration_time = None
        self.last_switch_time = -1e9

    def step(self, u_continuous):
        assert u_continuous >= 0.0 and u_continuous <= 1.0

        now = shared.simulation_time

        if self.last_integration_time is None:
            self.last_integration_time = now

        self.integrator += (u_continuous - self.output) * (now - self.last_integration_time)
        self.integrator = np.clip(self.integrator, -5, 5)  # Anti-Windup
        self.last_integration_time = now

        if (now - self.last_switch_time) >= self.min_hold_time:
            # keine Ahnung was korrekt ist - beides verhält sich ähnlich
            # Variante 1
            v = u_continuous + self.integrator
            # Variante 2
            # v = self.output + self.integrator

            idx = np.argmin(np.abs(self.levels - v))
            new_output = self.levels[idx]

            if new_output != self.output:
                self.output = new_output
                self.last_switch_time = shared.simulation_time

        return self.output