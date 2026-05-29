import numpy as np
import matplotlib.pyplot as plt
from controller import T_and_VPD_PID_Controller, PLGS_Controller
from utility import atmospheric_vpd_from_RH, RH_from_VPD_and_temp
from shared import DT
import shared


shared.simulation_time = 10000.0

pid_controller = T_and_VPD_PID_Controller(DT)
plgs_controller = PLGS_Controller(DT)

T_range = np.linspace(20, 30, 100)
RH_range = np.linspace(30, 80, 100)
u_h_p_map = np.full((100, 100), np.nan)
u_h_plgs_map = np.full((100, 100), np.nan)
u_f_p_map = np.full((100, 100), np.nan)
u_f_plgs_map = np.full((100, 100), np.nan)
vpd_map = np.full((100, 100), np.nan)
for iT, T in enumerate(T_range):
    for iRH, RH in enumerate(RH_range):
        shared.simulation_time += 100.0
        VPD = atmospheric_vpd_from_RH(T, RH)
        u_f_p, u_h_p, _, _, _ = pid_controller.get_control_variables(T, VPD)
        u_f_plgs, u_h_plgs = plgs_controller.get_control_variables(T, VPD, False)
        u_h_p_map[iT][iRH] = u_h_p
        u_h_plgs_map[iT][iRH] = u_h_plgs
        u_f_p_map[iT][iRH] = u_f_p
        u_f_plgs_map[iT][iRH] = u_f_plgs
        vpd_map[iT][iRH] = VPD


print(f"u_h_p    min={np.min(u_h_p_map)} max={np.max(u_h_p_map)}")
print(f"u_h_plgs min={np.min(u_h_plgs_map)} max={np.max(u_h_plgs_map)}")
print(f"u_f_p    min={np.min(u_f_p_map)} max={np.max(u_f_p_map)}")
print(f"u_f_plgs min={np.min(u_f_plgs_map)} max={np.max(u_f_plgs_map)}")

RH_from_T_lines = [RH_from_VPD_and_temp(1.1, T_range),
                   RH_from_VPD_and_temp(1.25, T_range),
                   RH_from_VPD_and_temp(1.4, T_range)]

fig, axs = plt.subplots(1, 5, figsize=(16, 6))

axs[0].imshow(u_h_p_map, origin='lower', aspect='auto',
           extent=[30, 80, 20, 30])
axs[0].set_title("u_h_p")
axs[0].set_xlabel("RH")
axs[0].set_ylabel("T")

axs[1].imshow(u_h_plgs_map, origin='lower', aspect='auto',
           extent=[30, 80, 20, 30])
axs[1].set_title("u_h_plgs")
axs[1].set_xlabel("RH")
axs[1].set_ylabel("T")

axs[2].imshow(u_f_p_map, origin='lower', aspect='auto',
           extent=[30, 80, 20, 30])
axs[2].set_title("u_f_p")
axs[2].set_xlabel("RH")
axs[2].set_ylabel("T")

axs[3].imshow(u_f_plgs_map, origin='lower', aspect='auto',
           extent=[30, 80, 20, 30])
axs[3].set_title("u_f_plgs")
axs[3].set_xlabel("RH")
axs[3].set_ylabel("T")

axs[4].imshow(vpd_map, origin='lower', aspect='auto',
           extent=[30, 80, 20, 30])
axs[4].set_title("VPD")
axs[4].set_xlabel("RH")
axs[4].set_ylabel("T")

for i_ax in range(5):
    for i_line in range(3):
        axs[i_ax].plot(RH_from_T_lines[i_line], T_range, color="green")
        axs[i_ax].plot(RH_range, np.ones_like(RH_range) * 25)
        axs[i_ax].plot(RH_range, np.ones_like(RH_range) * 28)

plt.tight_layout()
plt.show()