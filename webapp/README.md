# DiabIA - Interface Web

## Lancement

```bash
cd webapp
pip install flask tensorflow scikit-learn numpy
python app.py
```

Ouvrir : http://localhost:5000

## API

### POST /predict
Body JSON avec 21 features -> {probability, prediction, label, seuil, risque}

### GET /health
Retourne le statut du modele.
