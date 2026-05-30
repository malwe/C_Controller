import numpy as np

from utility import atmospheric_vpd_from_abs_hum


# Vereinfachungen

# - Luft im Zelt wird in Simulation als instantan vollständig durchmischt angenommen
#   -> Dadurch z.B. kein Temperatur-, Feuchte-, VPD-Gradient unten/oben
#      - Ausnahme: Bei Abluft wird Temperatur der abgesaugten Luft als T_i+T_gradient_offset angenommen
#   -> Kein 'Dynamisches Verhalten' bei Abluft und Humidifier
#   - Vernachlässigt weil: Schwierige Simulation

# - Abluft hat Soft Start durch Motorsteuerung
#   -> bei harten Sprüngen in u_f wird dynamisches Verhalten nicht modelliert
#   - Vernachlässigt weil: - Sind die Zeitkonstanten im Zelt nicht viel größer und damit der Effekt hier klein?
#                          - Unbekanntes Soft-Start-Verhalten

# - Abluft erhöht Luftbewegung im Zelt und damit Konvektion an Oberflächen
#   -> Kopplungen von Strukturen und Lufttem. nehmen zu
#   -> R_lambda nimmt ab
#   -> Transpiration nimmt zu (?)
#   -> Evaporatioin nimmt zu
#   -> tau_fog nimmt ab
#   - Vernachlässigt weil: Luftbewegung durch Abluft << Luftbewegung durch Ventilator

# - Ventilator auschaltbar (jetzt: immer an)
#   -> Kopplungen von Strukturen und Lufttemp. nehmen ab
#   -> R_lambda nimmt zu
#   -> Transpiration nimmt ab (?)
#   -> Evaporation nimmt ab
#   -> tau_fog nimmt zu
#   - Vernachlässigt weil: Bisher Ventilator immer an

# - Humidifier erzeugt Abwärme in Nebel, Luft und Wassertank
#   - Teilweise grob umgesetzt durch Q_hum_air = P_hum_air_factor * P_hum_max * u_h
#   - Der Rest müsste modelliert werden als übergehend in den Humidifier (Gerät+Wassertank)


class GrowZeltSim:
    def __init__(self, dt):
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
        self.P_hum_max = 25.0     # max el. Leistung des Humidifiers (W)

        # Modellparameter - später adaptive Schätzung

        # sehr grob:
        # R_lambda = R_h_i + R_zeltwand + R_h_a
        # R_lambda = 1/(h_i * A) + 1 / (U_zeltwand * A) + 1/(h_a * A)
        # A = 6.4 m^2 ohne Boden (Boden bisher nicht berücksichtigt)
        # h_i = 25 W/(m^2 K)
        # h_a = 3 W/(m^2 K)
        # U_zeltwand > 1000 W/(m^2 K) -> irrelevant
        # Falls Ventilator geschaltet werden soll muss R_lambda analog k_evap umgeschaltet werden
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
        self.eta_s_fast = 0.25    # LED-Leistungs Anteil Zeltwand, Pflanzen, Lampe selbst
        self.eta_s_slow = 0.05    # LED-Leistungs Anteil Wassertank, Erde
        assert(abs(self.eta_L + self.eta_s_fast + self.eta_s_slow - 1) < 1e-9)
  
        self.V_Lm = 0.05          # max Luftstrom (m^3/s) (nur Rohrventilator angeblich 0,096)
        self.n = 1.0              # Exponent Rohrventilator Kennlinie (vermutlich 1...3)

        # Altes Modell Transpiration
        # self.k_trans = 1.2e-5   # Transpiration pro VPD (kg/s/kPa) (hier 1kg/d bei 1kPa VPD)

        # Neues Modell Transpiration
        # Bisher keine Berücksichtigung der Windgeschwindigkeit (z.B. Ventilator {0, 1})
        self.tau_Gc = 20*60       # Dynamik der canopy conductance (s)
        self.Gc_ratio_no_light = 0.15  # (canopy conductance dunkel) / (canopy conductance hell) (1)
        self.K_half = 250         # (umol/m^2/s) (Bereich 200...300 gut belegt, am besten konstant lassen)
        self.PPFD_MAX = 700       # grob geschätzt (nochmal prüfen) (umol/m^2/s)
        self.Gc_max = 1e-5        # canopy conductance bei Gc_bezug (kg/s/kPa)
        self.Gc_VPD_g1 = 4        # VPD Abhängigkeit der canopy conductance (sqrt(kPa))
        Gc_bezugs_VPD = 1.0       # Bezugs-VPD für Initialisierung von Gc_max (kPa)
        self.Gc_norm = (1 + self.Gc_VPD_g1 / np.sqrt(Gc_bezugs_VPD))

        # https://www.sciencedirect.com/science/article/pii/S0304380025001188
        felt_evap_ratio = 0.15    # Verdunstung durch Topfmantel (bei Filz) vs. Verdunstung von Erdoberfläche (1)
        num_pots = 2              # Anzahl Pflanztöpfe (1)
        pot_diameter = 0.35       # (m)
        pot_fill_height = 0.25    # (m)
        pi = 3.14159
        earth_humidity = 0.5      # (0-1)
        A_evap = num_pots * (pi * (pot_diameter/2)**2 + felt_evap_ratio * pi * pot_diameter * pot_fill_height)
        # Modell lässt sich durch Umschalten von self.k_evap um Ventilator {0, 1} erweitern
        # Dazu muss z.B. Jacobian in EKF:
        # devap/dk_evap_vent_off = { vent=1: 0 ; vent=0: VPD }
        # devap/dk_evap_vent_on  = { vent=1: VPD ; vent=0: 0 }
        # h_v_vent_off = 3        # (mm/s) (2...5 bei fast keinem Wind)
        h_v_vent_on  = 12         # (mm/s) (ca. 20 bei überall 3 m/s Wind)
        # self.k_evap_vent_off = earth_humidity * h_v_vent_off * 7.26 * 10**(-7) * A_evap  # (kg/s/kPa)
        self.k_evap_vent_on  = earth_humidity * h_v_vent_on  * 7.26 * 10**(-7) * A_evap  # (kg/s/kPa)
        self.k_evap = self.k_evap_vent_on  # Später evtl. umschaltbar - Parameterschätzer auf k_evap_vent_on und k_evap_vent_off

        self.tau_fog = 5.0        # Dynamik (Nebler->) Nebel -> Dampf bei VPD=1kPa (s)

        self.P_hum_air_ratio = 0.5  # (0-1) Anteil der Leistung des Humidifiers welche direkt in die Luft abgegeben wird

        self.T_gradient_offset = 0.5  # Abluft saugt warme Luft oben aus Zelt: T_i+T_gradient_offset (K)

        # Turbulenzen (als Faktor): Ornstein-Uhlenbeck -> AR(1)
        tau_turb = 5.0       # (s) Zeitliche korrelation aus OU
        sigma = 0.05         # (1) Sigma des Turbulenzfaktors im stationären Zustand
        # AR(1)
        self.a = np.exp(-self.dt / tau_turb)
        self.sigma_eta = sigma * np.sqrt(1 - self.a**2)
        self.last_turb_s_fast = 0.0
        self.last_turb_s_slow = 0.0
        self.last_turb_wall = 0.0
        self.last_turb_evap = 0.0
        self.last_turb_trans = 0.0
        self.last_turb_fog_evap = 0.0

    # def rough_vpd_approx(self, T, m_H2O):
    #     # sehr vereinfachtes Modell (kein exakter Psychrometrie-Ansatz)
    #     RH = np.clip(m_H2O / (self.rho_L * self.V), 0.0, 1.0)
    #     return max(0.0, (1 - RH) * np.exp(0.05 * T))

    # Dynamik
    def step(self, x, u, d):
        T_i, T_s_slow, T_s_fast, m_H2O_vapor, Gc, m_fog = x  # (K) (K) (K) (kg) (kg/s/kPa) (kg)
        u_f, u_h = u                             # (0-1) (0-1)
        T_o, w_o, LED = d                        # (K) (kg/m^3) (0-1)

        # -------------------------
        # Hilfsgrößen
        # -------------------------
        Vdot = self.V_Lm * (u_f ** self.n)  # (m^3/s)

        # VPD = self.rough_vpd_approx(T_i, m_H2O)  # (kPa)
        VPD = max(0.0, atmospheric_vpd_from_abs_hum(T_i, m_H2O_vapor / self.V))  # (kPa)

        # REVIEW: ist Verdunstung der Nebeltropfen/ Aerosole so grob richtig skaliert?
        fog_evap_vpd_factor = VPD / 1.0  # (1) VPD skaliert die Nebelverdunstung
        # Turbulenzfaktor
        turb_fog_evap = self.turbulence(self.last_turb_fog_evap)
        self.last_turb_fog_evap = turb_fog_evap
        fog_evap = m_fog / self.tau_fog * fog_evap_vpd_factor * turb_fog_evap  # (kg/s)

        # Transpiration (kg/s) (der Pflanzen) ist canopy conductance (bezogen auf Growfläche) * VPD
        # Turbulenzfaktor
        turb_trans = self.turbulence(self.last_turb_trans)
        self.last_turb_trans = turb_trans
        trans = Gc * VPD * turb_trans

        # Verdunstung (kg/s) (von Erde und offenen Wasserflächen)
        # Turbulenzfaktor
        turb_evap = self.turbulence(self.last_turb_evap)
        self.last_turb_evap = turb_evap
        evap = self.k_evap * VPD * turb_evap

        # Turbulenzfaktor für s_slow
        turb_s_slow = self.turbulence(self.last_turb_s_slow)
        self.last_turb_s_slow = turb_s_slow

        # Turbulenzfaktor für s_fast
        turb_s_fast = self.turbulence(self.last_turb_s_fast)
        self.last_turb_s_fast = turb_s_fast

        # Turbulenzfaktor für Zeltwand
        turb_wall = self.turbulence(self.last_turb_wall)
        self.last_turb_wall = turb_wall

        # -------------------------
        # Zustand 1: Lufttemperatur im Zelt
        # -------------------------
        Q_vent = self.rho_L * Vdot * self.c_p * (T_o - (T_i + self.T_gradient_offset))  # (W)
        Q_wall = (T_o - T_i) / self.R_lambda * turb_wall
        Q_s_slow = self.k_s_slow * (T_s_slow - T_i) * turb_s_slow
        Q_s_fast = self.k_s_fast * (T_s_fast - T_i) * turb_s_fast
        Q_led_L = self.eta_L * LED * self.P_LED_max
        Q_fog_evap = - self.r * fog_evap
        Q_trans = - self.r * trans
        Q_evap = - self.r * evap
        Q_hum_air = self.P_hum_air_ratio * self.P_hum_max * u_h

        dT_i = (Q_vent + Q_wall + Q_s_slow + Q_s_fast + Q_led_L + Q_fog_evap + Q_trans + Q_evap + Q_hum_air) / (self.rho_L * self.V * self.c_p)

        # -------------------------
        # Zustand 2: Temperatur voluminöser wärmeträger Strukturen im Zelt (Wassertank, Erde)
        # -------------------------
        dT_s_slow = ( - Q_s_slow + self.eta_s_slow * LED * self.P_LED_max) / self.C_s_slow

        # -------------------------
        # Zustand 3: Temperatur flächiger wärmeträger Strukturen im Zelt (Zeltwände, Pflanzen, ...)
        # -------------------------
        dT_s_fast = ( - Q_s_fast + self.eta_s_fast * LED * self.P_LED_max) / self.C_s_fast

        # -------------------------
        # Zustand 4: Masse Wasserdampf im Zelt
        # -------------------------
        w_i = m_H2O_vapor / self.V  # absolute Feuchte (kg/m^3)
        dm_H2O_vapor = Vdot * (w_o - w_i) + m_fog / self.tau_fog + trans + evap  # (kg/s)

        # -------------------------
        # Zustand 5: Transpiration dynamisch
        # -------------------------
        # https://bg.copernicus.org/articles/21/1501/2024/
        # https://www.implexx.io/manuals-and-guides/how-to-measure-canopy-stomatal-conductance/
        # https://www.sciencedirect.com/science/article/pii/S0304380025001188
        # https://nph.onlinelibrary.wiley.com/doi/10.1111/nph.16485
        # (Selbst) modifiziertes Medlyn USO-Modell
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
        dm_fog = self.H * u_h - fog_evap - Vdot * m_fog / self.V  # (kg/s)

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

        return [[T_i, T_s_slow, T_s_fast, m_H2O_vapor, Gc, m_fog], [trans, evap]]  # x[k+1], abgeleitete_Groeßen[k+1]
    

    # Modell der Luftturbulenzen
    # Anwendung: Turbulenz als Faktor multiplizieren - z.B. v = v_mean * turb
    # Turbulenz als Ornstein-Uhlenbeck-Prozess modelliert
    # Überführung von kontinuierlichem OU zu diskretem AR(1):
    # OU: dturb = - 1/tau * turb * dt + sigma * dW_t
    # AR(1): x[k+1] = a*x[k] + N(0, sigma_eta^2)
    # sigma_eta = sigma_stationaer * sqrt(1 - a^2)
    # => x[k+1] = exp(-dt/tau) x[k] + N(0, sigma_eta^2)
    def turbulence(self, last_turb):
        turb = 1 + self.a * (last_turb - 1) + np.random.normal(0, self.sigma_eta)
        turb = np.clip(turb, 0.5, 1.5)
        return turb