import matplotlib.pyplot as plt
import numpy as np
from enum import Enum

from TentDynamics import GrowZeltSim
from SensorDynamics import SensorDynamics
from utility import RH_from_abs_hum, abs_hum_from_RH


DT = 0.2  # Zeitschritt der Simulation (s)

sim = GrowZeltSim(DT)

# tau_T_sensor und tau_RH_sensor frei geschätzt
tau_T_sensor = 15.0   # (s)
tau_RH_sensor = 15.0  # (s)
# EMA-Filter:
# y[i] = α x[i] + (1−α) y[i−1]
# Y(z) = α X(z) + (1−α) z^−1 Y(z)
# H(z) = Y(z)/X(z) = α / (1 - (1-α) z^-1)
# -> Pol: z_p = (1-α)
# Zeitdiskretes System mit Abtastzeit T_sampling: z_p = exp(-T_sampling / tau)
T_SensorSampling = 0.2  # (s)
alpha_EMA = 0.1         # (0-1)
tau_T_EMA = tau_RH_EMA = -T_SensorSampling / np.log(1-alpha_EMA)
sensor = SensorDynamics(DT, tau_T_sensor, tau_RH_sensor, tau_T_EMA, tau_RH_EMA)

# Initialzustand des Sensors im Zelt
sensor.init(T0=25.0, RH0=50.0)

# Initialzustand des Systems Zelt
x = np.array([25.0, 25.0, 15.0, 0.0, 0.0])  # T_i, T_s, m_H2O_vapor, trans, m_fog

# Zeitreihe
duration = 3600.0  # s
N = int(duration / DT)

# Stellgrößen (erstmal hardcoded, später Regelung)
# Abluft (0-1)
u_f = np.zeros(N)
u_f[N//8:N*5//8] = 0.5

# Humidifier (0-1)
u_h = np.zeros(N)
period = round(60 / DT)
for i in range(N//2, N - period, period):
    u_h[i:i + period // 2] = 1.0

# Störgrößen (erstmal hardcoded, später Messwerte)
T_o = np.ones(N) * 24.0   # (°C)
RH_o = np.ones(N) * 50.0  # (0-100 %)
w_o = abs_hum_from_RH(T_o, RH_o)  # absolute Feuchte der Außenluft (g/m^3)

P_LED = np.zeros(N)  # (0-1)
P_LED[N//4:N*3//4] = 1.0


# Bei Änderung auch Speichern der history unten anpassen
class HistoryRows(Enum):
    # Zustände
    T_i = 0
    T_s = 1
    m_H2O_vapor = 2
    trans = 3
    m_fog = 4
    # Stellgrößen
    u_f = 5
    u_h = 6
    # Sensorwerte (später evtl als Zustände modellieren)
    T_meas = 7
    RH_meas = 8
    # Abgeleitete Größen
    w_i = 9
    RH_i = 10
    #
    LENGTH_PLUS_ONE = 11

history = np.zeros((HistoryRows.LENGTH_PLUS_ONE.value, N))  # Zeilen: Historie der Größen, Spalten: Zeitschritte


# -------------------------
# Simulation
# -------------------------
for k in range(N):
    # Aktuelle Steuerungssignale und Störgrößen
    # für den Moment erstmal hardcoded, später Regelung und Messwerte
    u = [u_f[k], u_h[k]]
    z = [T_o[k], w_o[k], P_LED[k]]

    # Simulation eines Zeitschritts
    x = sim.step(x, u, z)
    T_i, T_s, m_H2O_vapor, trans, m_fog = x

    # Abgeleitete Größen für Plot
    w_i = m_H2O_vapor / sim.V  # absolute Feuchte (g/m^3)
    RH_i = RH_from_abs_hum(T_i, w_i)  # relative Feuchte (0-100 %)

    # Simulation der Sensorwerte mit Sensor-Dynamik
    T_meas, RH_meas = sensor.step(T_i, RH_i)

    # Historie speichern
    history[:, k] = [T_i, T_s, m_H2O_vapor, trans, m_fog, u_f[k], u_h[k], T_meas, RH_meas, w_i, RH_i]


# -------------------------
# Plot
# -------------------------
plt.figure()
plt.plot(np.arange(N) * DT, history[HistoryRows.T_i.value, :], label='T_i (K)')
plt.plot(np.arange(N) * DT, history[HistoryRows.T_s.value, :], label='T_s (K)')
# plt.plot(np.arange(N) * DT, history[HistoryRows.m_H2O.value, :], label='m_H2O (g)')
plt.plot(np.arange(N) * DT, history[HistoryRows.trans.value, :], label='trans (g/s)')
plt.plot(np.arange(N) * DT, history[HistoryRows.m_fog.value, :], label='m_fog (g)')
plt.plot(np.arange(N) * DT, history[HistoryRows.u_f.value, :], label='u_f (0-1)')
plt.plot(np.arange(N) * DT, history[HistoryRows.u_h.value, :], label='u_h (0-1)')
plt.plot(np.arange(N) * DT, history[HistoryRows.T_meas.value, :], label='T_meas (K)')
plt.plot(np.arange(N) * DT, history[HistoryRows.RH_meas.value, :], label='RH_meas (%)')
plt.plot(np.arange(N) * DT, history[HistoryRows.w_i.value, :], label='w_i (g/m^3)')
plt.plot(np.arange(N) * DT, history[HistoryRows.RH_i.value, :], label='RH_i (%)')

plt.plot(np.arange(N) * DT, T_o, label='T_o (K)', linestyle='--')
plt.plot(np.arange(N) * DT, RH_o, label='RH_o (%)', linestyle='--')
plt.plot(np.arange(N) * DT, w_o, label='w_o (g/m^3)', linestyle='--')
plt.plot(np.arange(N) * DT, P_LED, label='P_LED (0-1)', linestyle='--')
plt.legend()
plt.show()
