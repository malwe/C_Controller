# Cann_Controller

Ein kleines Grow-Zelt-Simulationsprojekt in Python.

## Inhalt

- `main.py` – Führt die Simulation aus, speichert historische Signale und plottet Ergebnisse.
- `GrowZeltSim.py` – Physikalisches Zeltmodell mit Luft-, Struktur- und Feuchtigkeitsdynamik.
- `SensorDynamics.py` – Sensorverzögerung und EMA-Filterung für Temperatur und Luftfeuchte.
- `utility.py` – Hilfsfunktionen zur Feuchteumrechnung und VPD-Berechnung.
- `requirements.txt` – Python-Abhängigkeiten.

## Voraussetzungen

- Python 3
- `numpy`
- `matplotlib`

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Ausführung

```bash
.venv/bin/python3 main.py
```

## Hinweise

- Die Simulation nutzt ein vereinfachtes physikalisches Modell und derzeit hardcodierte Stell- und Störgrößen.
