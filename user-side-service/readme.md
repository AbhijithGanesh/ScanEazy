# Team Plan

## Members
Neehal Reddy A -> 20BPS1151
Jaskaran Walia -> 21BCE1089
Abhijith Ganesh -> 21BRS1200

## Plan of Action
-> Machine Learning Path
-> Cloud Path
-> SBOM (Software Bill of Apps)
-> APIs

### Machine Learning

- Continuous ASCII.
- OCR classification
- Input : Scanned OMR As JSON
- Output : Identified answers as JSON

### API
Accepts JSON answers

- Cross references with answer key and evaluate scores
- Store Scores, Filled in Options in DB
- Connects with PSQL

### Cloud
- Runs the ML MODEL
- Runs the API Server
