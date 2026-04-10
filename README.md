# AutoTel AI — Telecom Intelligence Platform

## But de l'application
AutoTel AI est un dashboard de surveillance reseau telecom enterprise. Il permet aux equipes operationnelles de monitorer en temps reel l'etat de leur infrastructure reseau, de detecter automatiquement les anomalies grace au machine learning, de predire l'evolution du trafic et d'obtenir des recommandations d'optimisation. L'application centralise toutes les metriques critiques (debit, latence, congestion, signal, energie) en un seul outil, avec support de donnees reelles via import CSV.

---

## Fonctionnalites
- Tableau de bord temps reel (KPIs, trafic, latence, congestion)
- Surveillance multi-metriques (debit, pertes paquets, signal, utilisateurs)
- Carte geographique interactive des antennes (Abidjan)
- Import de donnees reelles CSV (SNMP, NetFlow, OSS/BSS)
- Prediction de trafic (Gradient Boosting)
- Detection d'anomalies (Isolation Forest)
- Optimisation et recommandations automatiques
- Simulateur de scenarios reseau
- Analyses intelligentes et alertes

---

## Installation

```bash
git clone https://github.com/votre-repo/autotel-ai.git
cd autotel-ai
pip install -r requirements.txt
streamlit run app.py
```

---

## Structure

```
autotel-ai/
├── app.py
├── requirements.txt
└── README.md
```

---

## Format CSV pour l'import

| Colonne         | Type     | Obligatoire |
|-----------------|----------|-------------|
| timestamp       | datetime | OUI         |
| traffic_gbps    | float    | OUI         |
| latency_ms      | float    | OUI         |
| congestion_pct  | float    | optionnel   |
| energy_kwh      | float    | optionnel   |
| packet_loss_pct | float    | optionnel   |
| throughput_mbps | float    | optionnel   |
| signal_dbm      | float    | optionnel   |
| active_users    | int      | optionnel   |

---

## Stack
- **Frontend** : Streamlit
- **Visualisation** : Plotly
- **ML** : scikit-learn (Isolation Forest, Gradient Boosting)
- **Data** : pandas, numpy

---

## Auteur
**ANOH AMON HEMERSON** — Data Scientist · Analytique Telecom · Systemes IA

*AutoTel AI v3.0 — Construit pour les operations telecom enterprise*
