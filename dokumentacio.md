# Wine Quality MLOps Projekt Dokumentáció

## 1. A projekt célja

A projekt célja egy teljes MLOps pipeline felépítése, amely magában foglalja:

- adatfeldolgozást  
- modell tanítást  
- modell regisztrációt  
- API szolgáltatást  
- drift monitoringot  
- workflow automatizálást Airflow segítségével  

---

## 2. Az adatkészlet

A Wine Quality dataset (WineQT.csv) borok kémiai tulajdonságait tartalmazza.  
A célváltozó a **quality** (0–10 skálán).

---

## 3. Modell tanítás

A modell tanítása:

- adat betöltése  
- train/test split  
- StandardScaler  
- RandomForestClassifier  
- MLflow loggolás  
- modell feltöltése Model Registry-be  

A tanítást Airflow-ból lehet automatizálni.

---

## 4. MLflow

MLflow feladatai:

- futások nyilvántartása  
- metrikák, paraméterek loggolása  
- modellek regisztrálása verziók szerint  
- modell kiszolgálása az API számára  

MLflow UI: **http://localhost:5000**

---

## 5. FastAPI

A REST API automatikusan betölti a Model Registry legfrissebb verzióját.

- végpont: `/predict`  
- input: JSON  
- output: előrejelzett minőség  

Swagger dokumentáció: **http://localhost:8000/docs**

---

## 6. Streamlit

A Streamlit szolgáltatás:

- interaktív predikciós UI  
- MLflow run lista  
- Evidently drift riport megjelenítése  

Elérés: **http://localhost:8501**

---

## 7. Drift monitoring – EvidentlyAI

A drift riportokat a következő paranccsal lehet generálni:

```bash
python src/wineclf/drift_report.py \
  --ref-data data/raw/WineQT.csv \
  --cur-data data/raw/WineQT.csv \
  --output-dir artifacts/evidently
```

A létrejött riport HTML formában tárolódik.

---

## 8. Airflow

Az Airflow orchestrálja az ML pipeline-t.

- **DAG neve:** `wine_training_pipeline`
- **Elérés:** http://localhost:8080

---

## 9. Docker Compose architektúra

A rendszer összes komponense konténerben fut:

- `mlflow` – modellkövetés és registry  
- `api` – FastAPI predikciós szolgáltatás  
- `streamlit` – UI alkalmazás  
- `airflow-scheduler`  
- `airflow-webserver`  
- `airflow-worker`  
- `postgres` – Airflow metadata DB  
- hálózat: `mlops_net`

---

## 10. Összefoglalás

A projekt egy teljes MLOps architektúrát demonstrál, amely:

- skálázható  
- verziózott  
- reprodukálható  
- bármikor újratanítható  
- valós időben szolgáltat predikciót  
- alkalmas drift monitoringra  

A rendszer production-ready alapot biztosít további fejlesztések számára.
