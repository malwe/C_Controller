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
        self.A = 0.64             # Zeltfläche (m^2)
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
        self.k_s_slow = 37.5           # Thermische Kopplung Luft <-> Struktur (W/K)

        # sehr grob:
        # C_s = m_s1 * c_s1 + m_s2 * c_s2 + ...
        # m_s1 = 12 kg (Wasser), c_s1 = 4200 J/(kg K)
        self.C_s_slow = 42000.0        # Wärmekapazität Struktur (J/K)

        # sehr grob:
        # k_s = 1/R_s = h_s * A_s
        # k_s = 25 W/(m^2 K)
        # A_s = 15 m^2  (Zeltwände, Pflanzen, flache/dünne Teile allg.)
        self.k_s_fast = 375           # Thermische Kopplung Luft <-> Struktur (W/K)

        # sehr grob:
        # C_s = m_s1 * c_s1 + m_s2 * c_s2 + ...
        # m_s1 = 1 kg (Wasser in Pflanzen), c_s1 = 4200 J/(kg K)
        # m_s2 = 1 kg (Aluminium), c_s2 = 900 J/(kg K)
        self.C_s_fast = 5100.0        # Wärmekapazität Struktur (J/K)
        
        self.eta_L = 0.7          # LED-Leistungs Anteil Luft
        self.eta_s_fast = 0.25    # LED-Leistungs Anteil Zeltwand, Pflanzen
        self.eta_s_slow = 0.05    # LED-Leistungs Anteil Wassertank, Erde
        assert(self.eta_L + self.eta_s_fast + self.eta_s_slow == 1.0)
  
        self.V_Lm = 0.05          # max Luftstrom (m^3/s) (nur Rohrventilator angeblich 0,096)
        self.n = 1.0              # Exponent Rohrventilator Kennlinie (vermutlich 1...3)

        # Altes Modell Transpiration
        # self.k_trans = 1.2e-5   # Transpiration pro VPD (kg/s/kPa) (hier 1kg/d bei 1kPa VPD)

        # Neues Modell Transpiration
        self.tau_Gc = 20*60       # Dynamik der canopy conductance (s)
        self.Gc_ratio_no_light = 0.15  # (canopy conductance dunkel) / (canopy conductance hell)
        self.K_half = 250         # (umol/m^2/s) (Bereich 200...300 gut belegt, am besten konstant lassen)
        self.PPFD_MAX = 700       # grob geschätzt (nochmal prüfen) (umol/m^2/s)
        self.Gc_max = 1e-5        # canopy conductance (bezogen auf self.A) (kg/s/kPa)
        self.Gc_VPD_g1 = 4        # VPD Abhängigkeit der canopy conductance (sqrt(kPa))
        Gc_bezug = 1.0
        self.Gc_norm = (1 + self.Gc_VPD_g1 / np.sqrt(Gc_bezug))

        self.tau_fog = 5.0        # Dynamik (Nebler->) Nebel -> Dampf bei VPD=1kPa(s)

        self.T_gradient_offset = 1.0   # Abluft saugt warme Luft oben aus Zelt: T_i+T_gradient_offset (K)

    # def rough_vpd_approx(self, T, m_H2O):
    #     # sehr vereinfachtes Modell (kein exakter Psychrometrie-Ansatz)
    #     RH = np.clip(m_H2O / (self.rho_L * self.V), 0.0, 1.0)
    #     return max(0.0, (1 - RH) * np.exp(0.05 * T))

    # Dynamik
    def step(self, x, u, z):
        T_i, T_s_slow, T_s_fast, m_H2O_vapor, Gc, m_fog = x  # (K) (K) (K) (kg) (kg/s/kPa) (kg)
        u_f, u_h = u                             # (0-1) (0-1)
        T_o, w_o, LED = z                        # (K) (kg/m^3) (0-1)

        # -------------------------
        # Hilfsgrößen
        # -------------------------
        Vdot = self.V_Lm * (u_f ** self.n)  # (m^3/s)

        # VPD = self.rough_vpd_approx(T_i, m_H2O)  # (kPa)
        VPD = max(0.0, leaf_vpd_from_abs_hum(T_i, m_H2O_vapor / self.V, LED))  # (kPa)

        # REVIEW: ist Verdunstung von VPD oder anderer Größe abhängig?
        fog_evap_factor = VPD / 1.0  # (1) VPD skaliert die Nebelverdunstung (kPa)

        # Verdunstung ist canopy conductance (bezogen auf Growfläche) * VPD
        trans = Gc * VPD

        # -------------------------
        # Zustand 1: Lufttemperatur im Zelt
        # -------------------------
        Q_vent = self.rho_L * Vdot * self.c_p * (T_o - (T_i + self.T_gradient_offset))  # (W)
        Q_wall = (T_o - T_i) / self.R_lambda
        Q_s_slow = self.k_s_slow * (T_s_slow - T_i)
        Q_s_fast = self.k_s_fast * (T_s_fast - T_i)
        Q_led_L = self.eta_L * LED * self.P_LED_max
        Q_hum = - self.r * m_fog / self.tau_fog * fog_evap_factor
        Q_trans = - self.r * trans

        dT_i = (Q_vent + Q_wall + Q_s_slow + Q_s_fast + Q_led_L + Q_hum + Q_trans) / (self.rho_L * self.V * self.c_p)

        # -------------------------
        # Zustand 2: Temperatur voluminöser wärmeträger Strukturen im Zelt (Wassertank, Erde)
        # -------------------------
        dT_s_slow = ( - self.k_s_slow * (T_s_slow - T_i) + self.eta_s_slow * LED * self.P_LED_max) / self.C_s_slow

        # -------------------------
        # Zustand 3: Temperatur flächiger wärmeträger Strukturen im Zelt (Zeltwände, Pflanzen, ...)
        # -------------------------
        dT_s_fast = ( - self.k_s_fast * (T_s_fast - T_i) + self.eta_s_fast * LED * self.P_LED_max) / self.C_s_fast

        # -------------------------
        # Zustand 4: Masse Wasserdampf im Zelt
        # -------------------------
        w_i = m_H2O_vapor / self.V  # absolute Feuchte (kg/m^3)
        dm_H2O_vapor = Vdot * (w_o - w_i) + m_fog / self.tau_fog + trans  # (kg/s)

        # -------------------------
        # Zustand 5: Transpiration dynamisch
        # -------------------------
        # https://bg.copernicus.org/articles/21/1501/2024/
        # https://www.implexx.io/manuals-and-guides/how-to-measure-canopy-stomatal-conductance/
        # https://www.sciencedirect.com/science/article/pii/S0304380025001188
        # https://nph.onlinelibrary.wiley.com/doi/10.1111/nph.16485
        PPFD = LED * self.PPFD_MAX  # Photosynthetically Active Photon Flux Density (µmol/m^2/s)
        Gc_VPD_abhaengigkeit = 1 + self.Gc_VPD_g1 / max(0.01, np.sqrt(VPD))
        Gc_steady = self.Gc_max/self.Gc_norm * (self.Gc_ratio_no_light + (1 - self.Gc_ratio_no_light) * PPFD / (PPFD + self.K_half)) * Gc_VPD_abhaengigkeit
        dGc = 1/self.tau_Gc * (Gc_steady - Gc)  # (kg/s/kPa/s)

        # Altes Modell PT1
        # dtrans = 1/self.tau_trans * (self.k_trans * VPD - trans)  # (kg/s/s)

        # -------------------------
        # Zustand 6: Nebel in Luft (noch als Tropfen, nicht als Dampf) - Annahme: PT1
        # -------------------------
        # -Vdot*m_fog/self.V -> Nebel wird durch Abluft entfernt
        # Praktisch ist die Abluft aber weit vom Nebel entfernt -> evtl. wieder entfernen?
        dm_fog = self.H * u_h - m_fog / self.tau_fog * fog_evap_factor - Vdot * m_fog / self.V  # (kg/s)

        # -------------------------
        # Euler Vorwärts-Integration
        # -------------------------
        T_i += self.dt * dT_i
        T_s_slow += self.dt * dT_s_slow
        T_s_fast += self.dt * dT_s_fast 
        m_H2O_vapor += self.dt * dm_H2O_vapor
        Gc += self.dt * dGc
        # trans += self.dt * dtrans
        m_fog += self.dt * dm_fog

        m_H2O_vapor = max(0.0, m_H2O_vapor)
        trans = max(0.0, trans)
        m_fog = max(0.0, m_fog)

        return np.array([T_i, T_s_slow, T_s_fast, m_H2O_vapor, Gc, m_fog])