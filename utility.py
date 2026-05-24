import numpy as np


# Blatt-Offset kann negativ oder positiv sein
# negativ: Verdunstungskühlung > LED-Erwärmung
# positiv: LED-Erwärmung > Verdunstungskühlung
leaf_offst_light_on =   0.0  # (K) bei 100% LED-Leistung (unklar -> messen!)
leaf_offset_light_off = 0.0  # (K) bei 0% LED-Leistung (sollte halbwegs stimmen)


# Atmosphärischer VPD (kPa) - Vorgaben

# https://www.cannabissciencetech.com/view/intensity-and-spectrum-the-role-of-lighting-in-vapor-pressure-deficit
atmospheric_vpd_vegetative = [-1000.0, 0.5, 0.7, 0.9, 1.1, 1.3, 1.7, 1000.0]  # (kPa)
atmospheric_vpd_flowering  = [-1000.0, 0.7, 1.0, 1.2, 1.5, 1.7, 2.3, 1000.0]  # (kPa)

# https://pmc.ncbi.nlm.nih.gov/articles/PMC12666426/pdf/fpls-16-1678142.pdf
# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10149009
# Atmosphärischer VPD (kPa)
# Seedling: 0.5-1.0 kPa (auch angegeben Soll: 75% RH)
# Vegetative: 0.7-1.2 kPa (auch angegeben Soll: 55-60% RH)
# Flowering: 1.0-1.5 kPa (auch angegeben Soll: 55-60% RH)

def _leaf_offset(P_LED_ratio):
    # Grobe Näherung: Je mehr LED-Power, desto höher die Blatttemperatur
    return leaf_offset_light_off + (leaf_offst_light_on - leaf_offset_light_off) * P_LED_ratio


# Relative humidity (0-100 %) from absolute humidity (g/m^3) and temperature (°C)
def RH_from_abs_hum(T, w):
    return w * (273.15 + T) / (6.112 * np.exp(17.67*T/(T+243.5)) * 2.1674)


# Absolute humidity (g/m^3) from relative humidity (0-100 %) and temperature (°C)
def abs_hum_from_RH(T, RH):
    return 6.112 * np.exp(17.67*T/(T+243.5)) * RH * 2.1674 / (273.15 + T)


# SVP - Sättigungsdampfdruck (kPa)
# https://pmc.ncbi.nlm.nih.gov/articles/PMC12666426/pdf/fpls-16-1678142.pdf
def _saturation_vapor_pressure(T):
    return 0.61078 * np.exp(17.2694*T/(T+238.3))


# Leaf Vapor Pressure Deficit (kPa)
def leaf_vpd_from_abs_hum(T, w, P_LED_ratio):
    RH = RH_from_abs_hum(T, w)
    return leaf_vpd_from_RH(T, RH, P_LED_ratio)


# Leaf Vapor Pressure Deficit (kPa)
def leaf_vpd_from_RH(T, RH, P_LED_ratio):
    SVP_air = _saturation_vapor_pressure(T)
    SVP_leaf = _saturation_vapor_pressure(T + _leaf_offset(P_LED_ratio))
    VPD_leaf = SVP_leaf - RH/100.0 * SVP_air
    return VPD_leaf


# Atmosphärischer VPD (kPa)
# https://pmc.ncbi.nlm.nih.gov/articles/PMC12666426/pdf/fpls-16-1678142.pdf
def atmospheric_vpd_from_RH(T, RH):
    SVP_air = _saturation_vapor_pressure(T)
    VPD_atm = (1 - RH/100.0) * SVP_air
    return VPD_atm
