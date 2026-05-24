import numpy as np

from utility import leaf_vpd_from_abs_hum


class GrowZeltSim:
    def __init__(self, dt=1.0):
        self.dt = dt

        # Physikalische Konstanten (linearisiert)
        self.rho_L = 1.23         # Dichte feuchter Luft(kg/m^3)
        self.c_p = 1005           # Spezifische Wärmekapazität von feuchter Luft (J/(kg K))
        self.r = 2450000.0        # Verdampfungsenthalpie bei 20°C (J/kg)

        # Konstante Modellparameter
        self.V = 1.15             # Zeltvolumen (m^3)
        self.H = 0.000061         # Humidifier (kg/s) max (220 g/h)
        self.V_Lm = 0.08          # max Luftstrom (m^3/s) (nur Rohrventilator angeblich 0,096)
        self.P_LED_max = 220.0    # max LED-Leistung (W)

        # Modellparameter - später adaptive Schätzung

        # sehr grob:
        # R_lambda = R_h_i + R_zeltwand + R_h_a
        # R_lambda = 1/(h_i * A) + 1 / (U_zeltwand * A) + 1/(h_a * A)
        # A = 6.4 m^2 ohne Boden (Boden bisher nicht berücksichtigt)
        # h_i = 25 W/(m^2 K)
        # h_a = 3 W/(m^2 K)
        # U_zeltwand > 1000 W/(m^2 K) -> irrelevant
        self.R_lambda = 0.06      # Wärmeübergang- & durchgang durch Zeltwand (K/W)

        # sehr grob:
        # k_s = 1/R_s = h_s * A_s
        # k_s = 25 W/(m^2 K)
        # A_s = 1.5 m^2
        self.k_s = 37.5           # Thermische Kopplung Luft <-> Struktur (W/K)

        # sehr grob:
        # C_s = m_s1 * c_s1 + m_s2 * c_s2 + ...
        # m_s1 = 1 kg (Aluminium), c_s1 = 900 J/(kg K)
        # m_s2 = 7 kg (Wasser), c_s2 = 4200 J/(kg K)
        self.C_s = 30000.0        # Wärmekapazität Struktur (J/K)
        
        self.eta_L = 0.7          # LED-Leistungs Anteil Luft; Rest: Strahlung -> Struktur
  
        self.V_Lm = 0.07          # max Luftstrom (m^3/s) (nur Rohrventilator angeblich 0,096)
        self.n = 1.0              # Exponent Rohrventilator Kennlinie (vermutlich 1...3)

        self.k_trans = 1e-6       # Transpiration scaling
        self.alpha_trans = 0.01   # Dynamik der Transpiration

        tau_fog = 5.0             # Dynamik Nebel -> Dampf (und Nebler) (s)
        self.k_fog = 1 / tau_fog  # Dynamik des Nebels (1/s))

    # def rough_vpd_approx(self, T, m_H2O):
    #     # sehr vereinfachtes Modell (kein exakter Psychrometrie-Ansatz)
    #     RH = np.clip(m_H2O / (self.rho_L * self.V), 0.0, 1.0)
    #     return max(0.0, (1 - RH) * np.exp(0.05 * T))

    # Dynamik
    def step(self, x, u, z):
        T_i, T_s, m_H2O_vapor, trans, m_fog = x  # (K) (K) (g) (g/s) (g)
        u_f, u_h = u                             # (0-1) (0-1)
        T_o, w_o, LED = z                        # (K) (g/m^3) (0-1)

        # -------------------------
        # Hilfsgrößen
        # -------------------------
        Vdot = self.V_Lm * (u_f ** self.n)  # m^3/s

        c_p = self.c_p

        # VPD = self.rough_vpd_approx(T_i, m_H2O)  # (kPa)
        VPD = leaf_vpd_from_abs_hum(T_i, m_H2O_vapor / self.V, LED)  # (kPa)

        d_eq = self.k_trans * VPD  # Transpirations-Gleichgewicht

        # -------------------------
        # Zustand 1: Lufttemperatur im Zelt
        # -------------------------
        Q_vent = self.rho_L * Vdot * c_p * (T_o - T_i)
        Q_wall = (T_o - T_i) / self.R_lambda
        Q_struct = self.k_s * (T_s - T_i)
        Q_led_L = self.eta_L * LED * self.P_LED_max
        Q_hum = - self.r * self.k_fog * m_fog
        Q_trans = - self.r * trans

        dT_i = (Q_vent + Q_wall + Q_struct + Q_led_L + Q_hum + Q_trans) / (self.rho_L * self.V * c_p)

        # -------------------------
        # Zustand 2: Temperatur wärmeträger Strukturen im Zelt
        # -------------------------
        dT_s = (-self.k_s * (T_s - T_i) + (1 - self.eta_L) * LED * self.P_LED_max) / self.C_s

        # -------------------------
        # Zustand 3: Masse Wasserdampf im Zelt
        # -------------------------
        w_i = m_H2O_vapor / self.V  # absolute Feuchte (g/m^3)
        dm = Vdot * (w_o - w_i) + self.k_fog * m_fog + trans  # (g/s)

        # -------------------------
        # Zustand 4: Transpiration dynamisch
        # -------------------------
        dd = -self.alpha_trans * (trans - d_eq)  # (g/s^2)

        # -------------------------
        # Zustand 5: Nebel in Luft (noch als Tropfen, nicht als Dampf) - Annahme: PT1
        # -------------------------
        dm_fog = self.H * u_h - self.k_fog * m_fog  # (g/s)

        # -------------------------
        # Euler Vorwärts-Integration
        # -------------------------
        T_i += self.dt * dT_i
        T_s += self.dt * dT_s
        m_H2O_vapor += self.dt * dm
        trans += self.dt * dd
        m_fog += self.dt * dm_fog

        return np.array([T_i, T_s, m_H2O_vapor, trans, m_fog])