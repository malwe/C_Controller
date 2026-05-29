import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from enum import Enum

from TentDynamics import GrowZeltSim
from SensorDynamics import SensorDynamics
from utility import RH_from_abs_hum, abs_hum_from_RH, atmospheric_vpd_from_RH, RH_from_VPD_and_temp
from controller import T_and_VPD_PID_Controller, PLGS_Controller, Bang_Bang_Controller
from shared import *
import shared


sim = GrowZeltSim(DT)

# tau_T_sensor und tau_RH_sensor frei geschätzt
tau_T_sensor =  10.0   # (s)
tau_RH_sensor = 10.0  # (s)
# EMA-Filter:
# y[i] = α x[i] + (1−α) y[i−1]
# Y(z) = α X(z) + (1−α) z^−1 Y(z)
# H(z) = Y(z)/X(z) = α / (1 - (1-α) z^-1)
# -> Pol: z_p = (1-α)
# Zeitdiskretes System mit Abtastzeit T_sampling: z_p = exp(-T_sampling / tau)
T_SensorSampling = DT   # (s) MUSS identisch sein mit DT !!!
alpha_EMA = 0.1         # (0-1)
tau_T_EMA = tau_RH_EMA = -T_SensorSampling / np.log(1-alpha_EMA)
# Sensorrauschen (frei geschätzt -> im Datenblatt suchen oder messen)
sigma_T  = 0.02  # (K)
sigma_RH = 0.1   # (%)
sensor = SensorDynamics(DT, tau_T_sensor, tau_RH_sensor, tau_T_EMA, tau_RH_EMA, sigma_T, sigma_RH)

# Initialzustand des Sensors im Zelt
sensor.init(T0=25.0, RH0=50.0)

# Initialzustand des Systems Zelt
x = np.array([25.0, 25.0, 25.0, 0.015, 0.0, 0.0])  # T_i, T_s_slow, T_s_fast, m_H2O_vapor(kg), Gc(kg/s/kPa), m_fog(kg)

# Zeitreihe
simulation_duration = 10 * 3600.0  # s
N = int(simulation_duration / DT)

# Störgrößen (erstmal hardcoded, später Messwerte & (Kalman-)Filter)
T_o = np.ones(N) * 24.0   # (°C)
RH_o = np.ones(N) * 50.0  # (0-100 %)
w_o = abs_hum_from_RH(T_o, RH_o)  # absolute Feuchte der Außenluft (kg/m^3)

LED = np.zeros(N)  # (0-1)
# LED[N//4:N*3//4] = 1.0
LED[N*1//3 : N*2//3] = 1.0


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
    trans = 14
    evap = 15
    # Reglerzustände/ -debugging
    D_f_T = 16
    D_f_vpd = 17
    D_h_vpd = 18
    #
    LENGTH_PLUS_ONE = 19

history = np.full((HistoryRows.LENGTH_PLUS_ONE.value, N), np.nan)  # Zeilen: Historie der Größen, Spalten: Zeitschritte


bang_bang_controller = Bang_Bang_Controller()
plgs_controller = PLGS_Controller()
pid_controller = T_and_VPD_PID_Controller(DT)

trans = float('NaN')
evap = float('NaN')

# -------------------------
# Simulation
# -------------------------
for k in range(N):
    shared.simulation_time = k * DT

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
    # u_f, u_h = bang_bang_controller.get_control_variables(T_meas, VPD_atm_meas)  # u[k]
    # u_f, u_h = plgs_controller.get_control_variables(T_meas, VPD_atm_meas, False)  # u[k]
    u_f, u_h = pid_controller.get_control_variables(T_meas, VPD_atm_meas, True)  # u[k]
    u = [u_f, u_h]

    D_f_T, D_f_vpd, D_h_vpd = pid_controller.get_D_filtered()

    history[:, k] = [T_i, T_s_slow, T_s_fast, m_H2O_vapor, Gc, m_fog, u_f, u_h, \
                     T_meas, RH_meas, w_i, RH_i, VPD_atm_i, VPD_atm_meas, trans, evap, D_f_T, D_f_vpd, D_h_vpd]

    # Simulation des nächsten Zeitschritts
    x, [trans, evap] = sim.step(x, u, z)  # x[k], u[k], z[k] -> x[k+1] 


# -------------------------
# Plot
# -------------------------
first_time_shown = 300.0  # (s)
first = round(first_time_shown / DT)

t = np.arange(N - first) * DT

# Zwei separate GridSpecs: links für 1D-Plots (75%), rechts für 2D-Plots (25%)
fig = plt.figure(figsize=(18, 12))

# Linke GridSpec: num_left_plots Zeilen für 1D-Plots
num_left_plots = 5
gs_left = fig.add_gridspec(num_left_plots, 1, left=0.03, right=0.72, top=0.98, bottom=0.03, 
                           hspace=0.07, height_ratios=[3, 3, 3, 3, 3],)

# Rechte GridSpec: 2 Zeilen für 2D-Plots (jeder bekommt volle Höhe)
gs_right = fig.add_gridspec(2, 1, left=0.75, right=0.99, top=0.95, bottom=0.4, 
                            hspace=0.15)

# 1D-Plots auf der linken Seite
axes_1d = []
for i in range(num_left_plots):
    if i == 0:
        axes_1d.append(fig.add_subplot(gs_left[i]))
    else:
        axes_1d.append(fig.add_subplot(gs_left[i], sharex=axes_1d[0]))

# 2D-Plots auf der rechten Seite
ax_2d_upper = fig.add_subplot(gs_right[0])
ax_2d_lower = fig.add_subplot(gs_right[1])

# Plot 1D: Steuergrößen u_f, u_h & Störgröße LED
axes_1d[0].plot(t, LED[first:], label='LED', color='tab:orange')
axes_1d[0].plot(t, history[HistoryRows.u_f.value, first:], label='u_f', color='tab:gray')
axes_1d[0].plot(t, history[HistoryRows.u_h.value, first:], label='u_h', color='tab:blue')
axes_1d[0].set_ylabel('Actuators & LED')
axes_1d[0].legend(loc='upper right', fontsize='small')
axes_1d[0].grid(True, linestyle=':', alpha=0.5)

# Plot 1D: Temperaturen
axes_1d[1].plot(t, history[HistoryRows.T_i.value, first:], label='T_i', color='tab:orange', linestyle='--')
axes_1d[1].plot(t, T_o[first:], label='T_o', color='tab:cyan', linestyle=':')
axes_1d[1].plot(t, history[HistoryRows.T_s_slow.value, first:], label='T_s_slow', color='tab:gray')
axes_1d[1].plot(t, history[HistoryRows.T_s_fast.value, first:], label='T_s_fast', color='tab:pink')
axes_1d[1].plot(t, history[HistoryRows.T_meas.value, first:], label='T_meas', color='tab:red')
axes_1d[1].plot(t, history[HistoryRows.D_f_T.value, first:], label='D_f_T', color='tab:green')
axes_1d[1].set_ylabel('Temperature (°C)')
axes_1d[1].legend(loc='upper right', fontsize='small')
axes_1d[1].grid(True, linestyle=':', alpha=0.5)

# Plot 1D: Relative Luftfeuchtigkeiten
axes_1d[2].plot(t, history[HistoryRows.RH_i.value, first:], label='RH_i', color='tab:orange', linestyle='--')
axes_1d[2].plot(t, RH_o[first:], label='RH_o', color='tab:cyan', linestyle=':')
axes_1d[2].plot(t, history[HistoryRows.RH_meas.value, first:], label='RH_meas', color='tab:red')
axes_1d[2].set_ylabel('RH (%)')
axes_1d[2].legend(loc='upper right', fontsize='small')
axes_1d[2].grid(True, linestyle=':', alpha=0.5)

# Plot 1D: Absolute Feuchten und Wassermassen
axes_1d[3].plot(t, w_o[first:], label='w_o (kg/m^3)', color='tab:cyan', linestyle=':')
axes_1d[3].plot(t, history[HistoryRows.w_i.value, first:], label='w_i (kg/m^3)', color='tab:blue')
axes_1d[3].plot(t, history[HistoryRows.m_fog.value, first:], label='m_fog (kg)', color='tab:brown')
axes_1d[3].plot(t, history[HistoryRows.trans.value, first:]*1000.0, label='trans (g/s)', color='tab:orange')
axes_1d[3].plot(t, history[HistoryRows.evap.value, first:]*1000.0, label='evap (g/s)', color='tab:pink')

axes_1d[3].set_ylabel('Humidity / mass')
axes_1d[3].legend(loc='upper right', fontsize='small')
axes_1d[3].grid(True, linestyle=':', alpha=0.5)

# Plot 1D: VPD und Gc
axes_1d[4].plot(t, history[HistoryRows.VPD_atm_i.value, first:], label='VPD_atm_i (kPa)', color='tab:orange', linestyle='--')
axes_1d[4].plot(t, history[HistoryRows.VPD_atm_meas.value, first:], label='VPD_atm_meas (kPa)', color='tab:red')
axes_1d[4].plot(t, history[HistoryRows.Gc.value, first:]*1000.0, label='Gc (g/s/kPa)', color='tab:cyan')
axes_1d[4].plot(t, history[HistoryRows.D_f_vpd.value, first:], label='D_f_vpd', color='tab:green')
axes_1d[4].plot(t, history[HistoryRows.D_h_vpd.value, first:], label='D_h_vpd', color='tab:blue')
axes_1d[4].set_ylabel('Other')
axes_1d[4].legend(loc='upper right', fontsize='small')
axes_1d[4].grid(True, linestyle=':', alpha=0.5)

axes_1d[num_left_plots - 1].set_xlabel('Time (s)')

# x-Achsen-Ziffern nur beim untersten Plot anzeigen
for i in range(num_left_plots - 1):
    axes_1d[i].tick_params(labelbottom=False)



# Grenzkurven für VPD berechnen
T_range = np.linspace(20, 30, 100)
RH_from_VPD_1_1  = RH_from_VPD_and_temp(1.1, T_range)
RH_from_VPD_1_2  = RH_from_VPD_and_temp(1.2, T_range)
RH_from_VPD_1_25 = RH_from_VPD_and_temp(1.25, T_range)
RH_from_VPD_1_3  = RH_from_VPD_and_temp(1.3, T_range)
RH_from_VPD_1_4  = RH_from_VPD_and_temp(1.4, T_range)

# Plot 2D oben: (RH_i, T_i)
RH_i_plot = history[HistoryRows.RH_i.value, first:]
T_i_plot = history[HistoryRows.T_i.value, first:]
heatmap, xedges, yedges = np.histogram2d(
    RH_i_plot,
    T_i_plot,
    bins=250,
    range=[[30, 80], [20, 30]]
)
heatmap = gaussian_filter(heatmap, sigma=5.0)
heatmap = heatmap / np.max(heatmap)
# alpha_map = heatmap.copy()
# alpha_map = alpha_map ** 0.5  # Gamma
ax_2d_upper.imshow(
    heatmap.T,
    origin='lower',
    extent=[30, 80, 20, 30],
    aspect='auto',
    cmap='Grays',
    alpha=1
)
line_2d_upper, = ax_2d_upper.plot(history[HistoryRows.RH_i.value, first:], history[HistoryRows.T_i.value, first:], 
                 color='tab:orange', linewidth=1.5, label='(RH_i, T_i)')
ax_2d_upper.plot(RH_from_VPD_1_1,   T_range, color='green', linestyle=':', linewidth=1.0, label=f'1.1')
ax_2d_upper.plot(RH_from_VPD_1_2, T_range, color='green', linestyle=':', linewidth=1.0, label=f'1.2')
ax_2d_upper.plot(RH_from_VPD_1_25,   T_range, color='green', linestyle=':', linewidth=1.0, label=f'1.25')
ax_2d_upper.plot(RH_from_VPD_1_3, T_range, color='green', linestyle=':', linewidth=1.0, label=f'1.3')
ax_2d_upper.plot(RH_from_VPD_1_4,   T_range, color='green', linestyle=':', linewidth=1.0, label=f'1.4')
ax_2d_upper.set_xlim(30, 80)
ax_2d_upper.set_ylim(20, 30)
ax_2d_upper.set_ylabel('T (°C)')
ax_2d_upper.set_title('RH_i vs T_i', fontsize='small', fontweight='bold')
ax_2d_upper.grid(True, linestyle=':', alpha=0.5)
ax_2d_upper.legend(fontsize='small', loc='upper left')

# Plot 2D unten: (RH_meas, T_meas)
RH_meas_plot = history[HistoryRows.RH_i.value, first:]
T_meas_plot = history[HistoryRows.T_i.value, first:]
heatmap, xedges, yedges = np.histogram2d(
    RH_meas_plot,
    T_meas_plot,
    bins=250,
    range=[[30, 80], [20, 30]]
)
heatmap = gaussian_filter(heatmap, sigma=5.0)
heatmap = heatmap / np.max(heatmap)
# alpha_map = heatmap.copy()
# alpha_map = alpha_map ** 0.5  # Gamma
ax_2d_lower.imshow(
    heatmap.T,
    origin='lower',
    extent=[30, 80, 20, 30],
    aspect='auto',
    cmap='Grays',
    alpha=1
)
line_2d_lower, = ax_2d_lower.plot(history[HistoryRows.RH_meas.value, first:], history[HistoryRows.T_meas.value, first:], 
                 color='tab:orange', linewidth=1.5, label='(RH_meas, T_meas)')
ax_2d_lower.plot(RH_from_VPD_1_1,  T_range, color='green', linestyle=':', linewidth=1.0, label=f'1.1')
ax_2d_lower.plot(RH_from_VPD_1_2,  T_range, color='green', linestyle=':', linewidth=1.0, label=f'1.2')
ax_2d_lower.plot(RH_from_VPD_1_25, T_range, color='green', linestyle=':', linewidth=1.0, label=f'1.25')
ax_2d_lower.plot(RH_from_VPD_1_3,  T_range, color='green', linestyle=':', linewidth=1.0, label=f'1.3')
ax_2d_lower.plot(RH_from_VPD_1_4,  T_range, color='green', linestyle=':', linewidth=1.0, label=f'1.4')
ax_2d_lower.set_xlim(30, 80)
ax_2d_lower.set_ylim(20, 30)
ax_2d_lower.set_xlabel('RH (%)')
ax_2d_lower.set_ylabel('T (°C)')
ax_2d_lower.set_title('RH_meas vs T_meas', fontsize='small', fontweight='bold')
ax_2d_lower.grid(True, linestyle=':', alpha=0.5)
ax_2d_lower.legend(fontsize='small', loc='upper left')

# Callback: Wenn x-Limits sich ändern, filtere die 2D-Plots
def on_xlim_change(event):
    xmin, xmax = axes_1d[0].get_xlim()
    
    # Konvertiere Zeit zu Indizes
    idx_min = int(max(0, xmin / DT)) + first
    idx_max = int(min(N, xmax / DT)) + first
    
    # Filtere Daten für den Bereich
    idx_slice = slice(idx_min, idx_max)
    
    # Update 2D oben (RH_i, T_i)
    line_2d_upper.set_data(
        history[HistoryRows.RH_i.value, idx_slice],
        history[HistoryRows.T_i.value, idx_slice]
    )
    
    # Update 2D unten (RH_meas, T_meas)
    line_2d_lower.set_data(
        history[HistoryRows.RH_meas.value, idx_slice],
        history[HistoryRows.T_meas.value, idx_slice]
    )
    
    fig.canvas.draw_idle()

# Registriere Callback
axes_1d[0].callbacks.connect('xlim_changed', on_xlim_change)

plt.show()
