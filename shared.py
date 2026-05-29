# Es muss mindestens gelten für Euler-Integration: DT / tau_min < 0.3
# EMA: T_sampling = 0.2 s ; alpha = 0.1 => tau_EMA = -T_sampling/ln(1 - alpha) = 1.9 s
# => DT < 0.57
DT = 0.20  # Zeitschritt der Simulation (s) MUSS identisch sein mit T_sampling der Sensoren !!!

# Einstellungen an Aktoren
U_F_ON = 0.35           # Eingestellte Lüfterleistung (0-1)
U_H_ON = 1.0           # Eingestellte Humidifier-Leistung (0-1)

# Regelvorgaben
VPD_MIN = 1.2          # (kPa)
VPD_MAX = 1.3          # (kPa)

T_MAX = 27.5           # (°C)
T_GOOD = 26.5          # (°C)


# Geteilte Variablen
simulation_time = 0.0  # (s)