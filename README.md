# Airline Review Rating Prediction – ML Projekt (Hausarbeit)

## Überblick

Dieses Repository enthält den vollständigen Code und die Auswertungen zur Hausarbeit über die Vorhersage von Airline-Bewertungen aus Freitexten (Airline Reviews).

Verglichen werden:
- klassische ML-Baselines
- Embedding-basierte Modelle
- ein DistilBERT-Regressor
- ein lokales Small-LLM (Phi-3 Mini) mit verschiedenen Output-Strategien (frei, JSON-Prompt, Schema-Prompt, Schema-Enforced, Repair)

Die Auswertung nutzt u. a.:
- MAE / RMSE / Spearman
- Coverage (Strict-Validität)
- Effizienzmetriken (`sec/100`)

---

## Ziel der Arbeit

Ziel ist die Analyse, wie gut unterschiedliche Modellfamilien numerische Ratings (1–10) aus Review-Texten vorhersagen können – mit besonderem Fokus auf:

- **Vorhersagequalität** (MAE, RMSE, Spearman)
- **Reliability / Validität strukturierter Outputs** (insbesondere bei LLMs)
- **Coverage im Strict-Setting**
- **Effizienz** (Laufzeit, `sec/100`)

---

## Inhalt des Repositories

### Wichtige Dateien

- `ML_Analysis_Project_Airline_Reviews.ipynb` – Hauptnotebook (komplette Pipeline)
- `Abgabe_Gilbert_Jack_ML_Hausarbeit_WS2025-26.pdf` – finale Hausarbeit
- `README.md` – diese Datei

### Automatisch erzeugte Ordner (durch das Notebook)

Das Notebook legt bei Ausführung automatisch eine Projektstruktur an:

- `data/raw/` – Rohdaten (CSV)
- `data/processed/` – bereinigte Daten + Splits
- `data/features/` – Embeddings/Feature-Caches
- `models/` – trainierte Modelle
- `reports/` – Metriken, Predictions, Tabellen, Exporte
- `figs/` – Abbildungen

---

## Datensatz

Verwendet wird der Airline-Reviews-Datensatz (CSV).

Das Notebook sucht die Datei in dieser Reihenfolge:

1. Umgebungsvariable `AIRLINE_CSV_PATH`
2. `data/raw/BA_AirlineReviews.csv`
3. `BA_AirlineReviews.csv` im Projektroot

### Erwarteter Dateiname (empfohlen)

`BA_AirlineReviews.csv`

### Hinweis zur Spaltenerkennung

Das Notebook erkennt Ziel- und Textspalten robust automatisch. Falls vorhanden, werden `ReviewHeader` + `ReviewBody` explizit als Textbasis verwendet (Titel + Body).

---

## Setup / Voraussetzungen

### Python

Empfohlen: **Python 3.10+**

### Pflicht-Pakete (Kernpipeline)

```bash
pip install numpy pandas scikit-learn pyarrow joblib matplotlib requests
```

### Für Embedding-Modell (MiniLM)

```bash
pip install torch transformers sentence-transformers
```

### Für DistilBERT-Regressor

```bash
pip install torch transformers
```

### Für LLM-Teil (Schema/JSON/Repair)

```bash
pip install requests jsonschema
```

Zusätzlich muss **Ollama** lokal laufen und das Modell **`phi3:mini`** installiert sein (für den LLM-Teil).

---

## Reproduzierbarkeit / Versionen

- Das Projekt nutzt Hugging Face `transformers`; in der Hausarbeit wird die Dokumentation auf die verwendete Version referenziert.
- Beim Start des Notebooks wird ein Environment-Snapshot nach `reports/environment.json` geschrieben.
- Ein `config.json` mit zentralen Parametern wird nach `reports/config.json` geschrieben.

---

## Ausführung (empfohlener Ablauf)

1. **CSV ablegen**  
   Lege `BA_AirlineReviews.csv` unter `data/raw/` ab  
   *(oder setze `AIRLINE_CSV_PATH`)*

2. **Notebook öffnen**  
   `ML_Analysis_Project_Airline_Reviews.ipynb`

3. **Zellen der Reihe nach ausführen**
   - Datenimport & Bereinigung
   - Splitting (train/val/test + subset `S`)
   - TF-IDF + Ridge
   - MiniLM + Ridge
   - DistilBERT-Regressor
   - Phi-3 Mini (LLM-Modi M1–M4)
   - Aggregation / Leaderboards / Grafiken / Exporttabellen

4. **Ergebnisse prüfen**
   - `reports/results.csv`
   - `reports/leaderboard_test.csv`
   - `reports/fairness_by_bin_test.csv`
   - `figs/*.png`

---

## Pipeline im Notebook

### 1) Datenvorbereitung
- Robustes CSV-Parsing (Delimiter/Encoding-Erkennung)
- Auswahl der Zielspalte (Rating)
- Auswahl/Erkennung der Textspalten
- Textbereinigung
- Filterung auf gültige Zielwerte `1..10`
- Speicherung als:
  - `data/processed/reviews_processed.parquet`
  - `data/processed/reviews_processed.csv`

### 2) Splits
- Stratified Split über Rating-Bins (`1–3`, `4–7`, `8–10`)
- Standardmäßig:
  - Train: 64%
  - Val: 16%
  - Test: 20%
- Zusätzlich: balanciertes Subset `S` (für LLM-Evaluation, standardmäßig bis zu 800 Beispiele)

Erzeugte Dateien:
- `data/processed/splits/train.parquet`
- `data/processed/splits/val.parquet`
- `data/processed/splits/test.parquet`
- `data/processed/splits/subset_S.parquet`

### 3) Modelle

#### A) TF-IDF + Ridge
- Grid-Search auf dem Validation-Split
- Finale Auswertung auf Test/Split `S`
- Speichert Modell als `.joblib`

#### B) MiniLM + Ridge
- `all-MiniLM-L6-v2` Embeddings (mit Cache)
- Ridge-Regressor auf Embeddings
- Embeddings werden als `.npz` gespeichert (wiederverwendbar)

#### C) DistilBERT-Regressor
- `distilbert-base-uncased`
- Regressionskopf (`num_labels=1`)
- Early Stopping + gespeicherte Modellartefakte im `models/`-Ordner

#### D) Phi-3 Mini (lokal via Ollama)
LLM-Modi u. a.:
- `M1_FREE`
- `M2_JSON_PROMPT`
- `M3_SCHEMA_PROMPT`
- `M3_SCHEMA_ENFORCED`
- `M4_REPAIR`

Fokus: Reliability/Validität strukturierter Ausgaben (JSON/Schema) und Coverage im Strict-Setting.

---

## Wichtige Output-Dateien

### `reports/`
Typische Ausgaben (je nach ausgeführten Notebook-Teilen):

- `config.json`
- `environment.json`
- `data_schema.json`
- `splits_meta.json`
- `subset_S_meta.json`
- `results.csv`
- `predictions_*.parquet`
- `leaderboard_test.csv`
- `leaderboard_S.csv`
- `fairness_by_bin_test.csv`
- `bootstrap_ci_test_strict.csv`
- `bootstrap_delta_mae_test_strict.csv`
- `error_analysis_summary.csv`
- `artifact_index.csv`
- `llm_cache*.jsonl`
- `reports/runs/...` (Sweep-/Run-spezifische Outputs)
- `paper_exports_*` (Tabellen-/Abbildungs-Exporte für das Paper)

### `figs/`
Beispielhafte Grafiken:
- `leaderboard_mae_test.png`
- `leaderboard_coverage_test.png`
- `coverage_vs_mae_test.png`
- `efficiency_vs_mae_test.png`
- `*_pred_vs_true_test.png` (pro Modell)

---

## Interpretation der Metriken (kurz)

- **MAE / RMSE**: kleiner = besser
- **Spearman**: größer = besser
- **Coverage (strict)**: Anteil vollständig valider Outputs (v. a. relevant für LLMs)
- **sec/100**: Laufzeit normalisiert auf 100 Beispiele (kleiner = besser)

---

## Abgabehinweise

### Muss ich den Code abgeben?
Empfehlung: **Ja** (sehr sinnvoll für Nachvollziehbarkeit / Reproduzierbarkeit).

### Empfohlener Abgabeumfang (Repo)
**Abgeben:**
- `ML_Analysis_Project_Airline_Reviews.ipynb`
- `README.md`
- `Abgabe_Gilbert_Jack_ML_Hausarbeit_WS2025-26.pdf`
- optional: zentrale Ergebnisdateien (`reports/*.csv`, `figs/*.png`)
- optional: `requirements.txt` (sehr empfehlenswert)

**Nicht zwingend abgeben (nur wenn gefordert):**
- große Modellgewichte (`models/distilbert_*`)
- große Embedding-Caches (`data/features/*.npz`)
- große LLM-Caches (`reports/llm_cache*.jsonl`)
- temporäre Run-Ordner / Zwischenstände

---

## Empfohlene `requirements.txt` (optional)

Für eine saubere Reproduktion kannst du lokal ausführen:

```bash
pip freeze > requirements.txt
```

Wenn du eine schlanke Variante willst, reicht auch eine manuell gepflegte Datei mit den Kernpaketen.

---

## Troubleshooting

### CSV wird nicht gefunden
Fehlermeldung wie „CSV nicht gefunden …“  
→ Datei nach `data/raw/BA_AirlineReviews.csv` legen oder:

```bash
export AIRLINE_CSV_PATH="/voller/pfad/zu/BA_AirlineReviews.csv"
```

### Ollama nicht erreichbar
Wenn der LLM-Teil fehlschlägt:
- prüfen, ob Ollama läuft
- prüfen, ob `phi3:mini` installiert ist
- Notebook-Zellen für LLM ggf. überspringen (wenn nur Baselines reproduziert werden sollen)

### `pyarrow` fehlt (Parquet-Fehler)

```bash
pip install pyarrow
```

### `jsonschema` fehlt (LLM Schema/Repair)

```bash
pip install jsonschema
```

---

## Autor

Jack Gilbert  
Hausarbeit / ML-Projekt, WS 2025/26

---

## Lizenz / Nutzung

Nur für Studien-/Abgabezwecke im Rahmen der Lehrveranstaltung.  
(Bei Bedarf hier eine offizielle Lizenz ergänzen, z. B. MIT.)
