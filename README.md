# AutoTel AI ??? Telecom Intelligence Platform

## But de l'application
AutoTel AI est un dashboard de surveillance r??seau t??l??com enterprise. Il permet aux ??quipes op??rationnelles de monitorer en temps r??el l'??tat de leur infrastructure r??seau, de d??tecter automatiquement les anomalies gr??ce au machine learning, de pr??dire l'??volution du trafic et d'obtenir des recommandations d'optimisation. L'application centralise toutes les m??triques critiques (d??bit, latence, congestion, signal, ??nergie) en un seul outil, avec support de donn??es r??elles via import CSV.

---

## Fonctionnalit??s
- ???? Tableau de bord temps r??el (KPIs, trafic, latence, congestion)
- ???? Surveillance multi-m??triques (d??bit, pertes paquets, signal, utilisateurs)
- ??????? Carte g??ographique interactive des antennes (Abidjan)
- ???? Import de donn??es r??elles CSV (SNMP, NetFlow, OSS/BSS)
- ???? Pr??diction de trafic (Gradient Boosting)
- ???? D??tection d'anomalies (Isolation Forest)
- ?????? Optimisation et recommandations automatiques
- ??????? Simulateur de sc??narios r??seau
- ???? Analyses intelligentes et alertes

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
????????? app.py
????????? requirements.txt
????????? README.md
```

---

## Format CSV pour l'import
| Colonne | Type | Obligatoire |
|---|---|---|
| timestamp | datetime | ??? |
| traffic_gbps | float | ??? |
| latency_ms | float | ??? |
| congestion_pct | float | ??? |
| energy_kwh | float | ??? |
| packet_loss_pct | float | ??? |
| throughput_mbps | float | ??? |
| signal_dbm | float | ??? |
| active_users | int | ??? |

---

## Stack
- **Frontend** : Streamlit
- **Visualisation** : Plotly
- **ML** : scikit-learn (Isolation Forest, Gradient Boosting)
- **Data** : pandas, numpy

---

## Auteur
**ANOH AMON HEMERSON** ??? Data Scientist ?? Analytique T??l??com ?? Syst??mes IA

*AutoTel AI v3.0 ??? Construit pour les op??rations t??l??com enterprise*