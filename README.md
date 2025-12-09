# Wine Quality ML Pipeline – MLOps Projekt

Ez a projekt egy teljes MLOps rendszert valósít meg a Wine Quality adathalmazon.  
A rendszer célja egy production-ready architektúra bemutatása: modell tanítás, verziózás, API szolgáltatás, drift monitoring és pipeline-automatizálás.

---

## 🚀 Fő komponensek

- **MLflow** – modellkövetés, Model Registry  
- **FastAPI** – predikciós REST API  
- **Streamlit** – webes UI és dashboard  
- **EvidentlyAI** – data drift riportok  
- **Apache Airflow** – ML pipeline orchestration  
- **Docker Compose** – konténerizált futtatási környezet  

---

## 🧰 Használt technológiák

Python, scikit-learn, MLflow, FastAPI, Streamlit, EvidentlyAI, Airflow, Docker, Docker Compose

---

## ⚙️ A rendszer indítása

A teljes infrastruktúra elindítása:

```bash
docker compose up --build

```
## Elérhető szolgáltatások

- **MLflow UI:** http://localhost:5000  
- **Streamlit UI:** http://localhost:8501  
- **FastAPI / Swagger:** http://localhost:8000/docs  
- **Airflow UI:** http://localhost:8080  

---

## Modell tanítása Airflow segítségével

A modell újratanítása az Airflow UI felületéről indítható.

- **DAG neve:** `wine_training_pipeline`

### A pipeline lépései:

1. Adatok betöltése  
2. Modell tanítása  
3. Metrikák és paraméterek loggolása MLflow-ba  
4. Új modellverzió regisztrálása  
5. Drift riport készítése Evidently-vel  

### Evidently riport generálása (parancssorból):

```bash
python src/wineclf/drift_report.py \
  --ref-data data/raw/WineQT.csv \
  --cur-data data/raw/WineQT.csv \
  --output-dir artifacts/evidently
```

A riportot a Streamlit automatikusan felismeri és megjeleníti, ha létezik.

---

## REST API használata

A Swagger UI elérhető itt:  
**http://localhost:8000/docs**

### Példa request:

```json
{
  "fixed_acidity": 7.1,
  "volatile_acidity": 0.52,
  "citric_acid": 0.04,
  "residual_sugar": 1.8,
  "chlorides": 0.078,
  "free_sulfur_dioxide": 20,
  "total_sulfur_dioxide": 65,
  "density": 0.9972,
  "pH": 3.41,
  "sulphates": 0.61,
  "alcohol": 10.4
}
```

## GitHubra feltöltendő fájlok

- `src/`
- `airflow/dags/`
- `docker-compose.yml`
- `Dockerfile`
- `Dockerfile.airflow`
- `streamlit_app.py`
- `requirements.txt`
- `README.md`
- `dokumentacio.md`

---

## GitHub-ra **nem** kerülnek fel

- `.venv/`
- `__pycache__/`
- `mlflow/`
- `artifacts/`
- `airflow/logs/`
- SQLite adatbázis fájlok (`*.db`, `*.sqlite`)

---

## Összegzés

A projekt egy modern, production-közeli MLOps rendszert valósít meg, amely:

- automatizálja a modell tanítását  
- verziókezeli és regisztrálja a modelleket  
- REST API-n keresztül szolgáltat predikciót  
- drift monitoringot biztosít Evidently segítségével  
- teljesen konténerizált Docker Compose környezetben fut  

A rendszer alkalmas további bővítésre, CI/CD integrációra és felhős deploymentre.