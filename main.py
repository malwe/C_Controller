import matplotlib.pyplot as plt
import numpy as np
from enum import Enum

from TentDynamics import GrowZeltSim
from SensorDynamics import SensorDynamics
from utility import RH_from_abs_hum, abs_hum_from_RH, atmospheric_vpd_from_RH
from controller import Controller


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
x = np.array([25.0, 25.0, 25.0, 0.015, 0.0, 0.0])  # T_i, T_s_slow, T_s_fast, m_H2O_vapor(kg), Gc(kg/s/kPa), m_fog(kg)

# Zeitreihe
simulation_duration = 8 * 3600.0  # s
N = int(simulation_duration / DT)

# Stellgrößen (erstmal hardcoded, später Regelung)
# Abluft (0-1)
u_f = np.zeros(N)
# u_f[N//8:N*5//8] = 0.2
for i in range(round(N/2), N-round(120/DT), round(120/DT)):
    u_f[i:i + round(20/DT)] = 0.25

# Humidifier (0-1)
u_h = np.zeros(N)
period = round(120 / DT)
for i in range(round(N/2+period/2), N - 2*period, period):
    u_h[i:i + period // 3] = 1.0

# Störgrößen (erstmal hardcoded, später Messwerte)
T_o = np.ones(N) * 24.0   # (°C)
RH_o = np.ones(N) * 50.0  # (0-100 %)
w_o = abs_hum_from_RH(T_o, RH_o)  # absolute Feuchte der Außenluft (kg/m^3)

LED = np.zeros(N)  # (0-1)
# LED[N//4:N*3//4] = 1.0
LED[N*1//3:N*3//4] = 1.0


# Bei Änderung auch Speichern der history unten anpassen
class HistoryRows(Enum):
    # Zustände
    T_i = 0
    T_s_slow = 1
    T_s_fast = 2
    m_H2O_vapor = 3
    Gc = 4
    m_fog = 5
    # Stellgrößen
    u_f = 6
    u_h = 7
    # Sensorwerte (später evtl als Zustände modellieren)
    T_meas = 8
    RH_meas = 9
    # Abgeleitete Größen
    w_i = 10
    RH_i = 11
    VPD_atm_i = 12
    VPD_atm_meas = 13
    #
    LENGTH_PLUS_ONE = 14

history = np.zeros((HistoryRows.LENGTH_PLUS_ONE.value, N))  # Zeilen: Historie der Größen, Spalten: Zeitschritte


controller = Controller(DT)

# -------------------------
# Simulation
# -------------------------
for k in range(N):
    # Störgrößen (erstmal hardcoded, später Messwerte)
    z = [T_o[k], w_o[k], LED[k]]  # z[k]

    T_i, T_s_slow, T_s_fast, m_H2O_vapor, Gc, m_fog = x

    # Abgeleitete Größen für Plot
    w_i = m_H2O_vapor / sim.V  # absolute Feuchte (kg/m^3)
    RH_i = RH_from_abs_hum(T_i, w_i)  # relative Feuchte (0-100 %)

    # Simulation der Sensorwerte mit Sensor-Dynamik
    T_meas, RH_meas = sensor.step(T_i, RH_i)

    VPD_atm_i = atmospheric_vpd_from_RH(T_i, RH_i)
    VPD_atm_meas = atmospheric_vpd_from_RH(T_meas, RH_meas)

    # Stellgrößen (hardcoded)
    # u = [u_f[k], u_h[k]]
    # Stellgrößen (Regler)
    simulation_time = k * DT
    u = controller.compute_control(simulation_time, T_meas, VPD_atm_meas)  # u[k]
    u_f[k], u_h[k] = u

    # Historie speichern
    history[:, k] = [T_i, T_s_slow, T_s_fast, m_H2O_vapor, Gc, m_fog, u_f[k], u_h[k], T_meas, RH_meas, w_i, RH_i, VPD_atm_i, VPD_atm_meas]

    # Simulation des nächsten Zeitschritts
    x = sim.step(x, u, z)  # x[k], u[k], z[k] -> x[k+1] 


# -------------------------
# Plot
# -------------------------
t = np.arange(N) * DT
fig, axes = plt.subplots(
    5,
    1,
    sharex=True,
    figsize=(14, 9),
    gridspec_kw={"height_ratios": [1, 3, 3, 3, 3], "hspace": 0.05},
)
fig.subplots_adjust(left=0.05, right=0.99, top=0.98, bottom=0.05)

# Plot 1: Steuergrößen u_f, u_h & Störgröße LED
axes[0].plot(t, history[HistoryRows.u_f.value, :], label='u_f', color='tab:blue')
axes[0].plot(t, history[HistoryRows.u_h.value, :], label='u_h', color='tab:orange')
axes[0].plot(t, LED, label='LED', color='tab:green')
axes[0].set_ylabel('Actuators & LED')
axes[0].legend(loc='upper right')
axes[0].grid(True, linestyle=':', alpha=0.5)

# Plot 2: Temperaturen
axes[1].plot(t, history[HistoryRows.T_i.value, :], label='T_i', color='tab:red', linestyle='--')
axes[1].plot(t, T_o, label='T_o', color='tab:cyan', linestyle=':')
axes[1].plot(t, history[HistoryRows.T_s_slow.value, :], label='T_s_slow', color='tab:purple')
axes[1].plot(t, history[HistoryRows.T_s_fast.value, :], label='T_s_fast', color='tab:pink')
axes[1].plot(t, history[HistoryRows.T_meas.value, :], label='T_meas', color='tab:gray')
axes[1].set_ylabel('Temperature (°C)')
axes[1].legend(loc='upper right')
axes[1].grid(True, linestyle=':', alpha=0.5)

# Plot 3: Relative Luftfeuchtigkeiten
axes[2].plot(t, history[HistoryRows.RH_i.value, :], label='RH_i', color='tab:blue', linestyle='--')
axes[2].plot(t, RH_o, label='RH_o', color='tab:cyan', linestyle=':')
axes[2].plot(t, history[HistoryRows.RH_meas.value, :], label='RH_meas', color='tab:orange')
axes[2].set_ylabel('RH (%)')
axes[2].legend(loc='upper right')
axes[2].grid(True, linestyle=':', alpha=0.5)

# Plot 4: Absolute Feuchten und Wassermassen
axes[3].plot(t, w_o, label='w_o', color='tab:cyan', linestyle='--')
axes[3].plot(t, history[HistoryRows.w_i.value, :], label='w_i', color='tab:blue')
axes[3].plot(t, history[HistoryRows.m_H2O_vapor.value, :], label='m_H2O_vapor', color='tab:green')
axes[3].plot(t, history[HistoryRows.m_fog.value, :], label='m_fog', color='tab:brown')
axes[3].plot(t, history[HistoryRows.Gc.value, :]*history[HistoryRows.VPD_atm_i.value, :]*1000.0, label='trans (g/s)', color='tab:orange')
axes[3].set_ylabel('Humidity (kg/m^3) / mass (kg)')
axes[3].set_xlabel('Time (s)')
axes[3].legend(loc='upper right')
axes[3].grid(True, linestyle=':', alpha=0.5)

# Plot 4: Sonstiges
axes[4].plot(t, history[HistoryRows.VPD_atm_i.value, :], label='VPD_atm_i', color='tab:gray', linestyle='--')
axes[4].plot(t, history[HistoryRows.VPD_atm_meas.value, :], label='VPD_atm_meas', color='tab:blue')
axes[4].plot(t, history[HistoryRows.Gc.value, :]*1000.0, label='Gc (g/s/kPa)', color='tab:orange')
axes[4].set_ylabel('Other')
axes[4].set_xlabel('Time (s)')
axes[4].legend(loc='upper right')
axes[4].grid(True, linestyle=':', alpha=0.5)

plt.show()
