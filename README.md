# 🎓 PlacementAI: Campus Placement Predictor

A modern web app that predicts whether a student is likely to be **placed** in campus recruitment, using a trained **Decision Tree** classifier (scikit-learn). Built with Flask and a single-file glassmorphism UI, and ready to deploy on Vercel.

## ✨ Features

- Dark, glassmorphism UI with animated gradient background
- Interactive sliders for scores and segmented toggles for categories
- Animated result ring with a Placed / Not Placed verdict
- Server-side input validation
- REST API endpoint (`/api/predict`) plus a health check
- Fully responsive (desktop, tablet, mobile)
- Entire app in a single `app.py` file

## 🧰 Tech Stack

| Layer      | Technology                       |
|------------|----------------------------------|
| Backend    | Python, Flask                    |
| ML Model   | scikit-learn `DecisionTreeClassifier` |
| Numerics   | NumPy                            |
| Frontend   | HTML, CSS, vanilla JavaScript (inline in `app.py`) |
| Deployment | Vercel                           |

## 📁 Project Structure

```
.
├── app.py             # Flask backend + complete UI
├── decision.pkl       # Trained decision tree model
├── requirements.txt   # Python dependencies
└── README.md
```

## 📊 Model Input Features

The model expects 12 features, in this order:

| Feature          | Description                    | Accepted values                          |
|------------------|--------------------------------|------------------------------------------|
| `gender`         | Gender                         | `M`, `F`                                 |
| `ssc_p`          | 10th grade percentage          | 0 – 100                                  |
| `ssc_b`          | 10th grade board               | `Central`, `Others`                      |
| `hsc_p`          | 12th grade percentage          | 0 – 100                                  |
| `hsc_b`          | 12th grade board               | `Central`, `Others`                      |
| `hsc_s`          | 12th grade stream              | `Arts`, `Commerce`, `Science`            |
| `degree_p`       | Degree percentage              | 0 – 100                                  |
| `degree_t`       | Degree type                    | `Comm&Mgmt`, `Sci&Tech`, `Others`        |
| `workex`         | Work experience                | `Yes`, `No`                              |
| `etest_p`        | Employability test percentage  | 0 – 100                                  |
| `specialisation` | MBA specialisation             | `Mkt&Fin`, `Mkt&HR`                      |
| `mba_p`          | MBA percentage                 | 0 – 100                                  |

### Categorical encoding

Categories are label-encoded alphabetically (same as scikit-learn's `LabelEncoder`):

| Feature          | Encoding                                         |
|------------------|--------------------------------------------------|
| `gender`         | F=0, M=1                                         |
| `ssc_b`, `hsc_b` | Central=0, Others=1                              |
| `hsc_s`          | Arts=0, Commerce=1, Science=2                    |
| `degree_t`       | Comm&Mgmt=0, Others=1, Sci&Tech=2                |
| `workex`         | No=0, Yes=1                                      |
| `specialisation` | Mkt&Fin=0, Mkt&HR=1                              |

> If you retrain the model with a different encoding, update the `ENCODERS` dictionary in `app.py`.

## 🚀 Run Locally

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd <your-repo-folder>

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the app
python app.py
```

Open **http://localhost:5000** in your browser.

## ☁️ Deploy on Vercel

**Option 1: Vercel Dashboard**

1. Push this project to a GitHub repository.
2. Go to [vercel.com](https://vercel.com) → **Add New → Project**.
3. Import your repository and click **Deploy**.

**Option 2: Vercel CLI**

```bash
npm i -g vercel
vercel --prod
```

Vercel detects the Flask `app` object in `app.py` automatically, so no extra config is needed.

> ⚠️ Keep `decision.pkl` in the project root, next to `app.py`.

## 🔌 API Reference

### `POST /api/predict`

**Request body (JSON)**

```json
{
  "gender": "M",
  "ssc_p": 80,
  "ssc_b": "Central",
  "hsc_p": 75,
  "hsc_b": "Others",
  "hsc_s": "Commerce",
  "degree_p": 72,
  "degree_t": "Comm&Mgmt",
  "workex": "Yes",
  "etest_p": 80,
  "specialisation": "Mkt&Fin",
  "mba_p": 70
}
```

**Success response**

```json
{
  "ok": true,
  "prediction": "Placed",
  "placed": true,
  "placed_probability": 100.0
}
```

**Error response (HTTP 400)**

```json
{
  "ok": false,
  "error": "Invalid input: ssc_p must be between 0 and 100"
}
```

**Example with cURL**

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"gender":"M","ssc_p":80,"ssc_b":"Central","hsc_p":75,"hsc_b":"Others","hsc_s":"Commerce","degree_p":72,"degree_t":"Comm&Mgmt","workex":"Yes","etest_p":80,"specialisation":"Mkt&Fin","mba_p":70}'
```

### `GET /health`

Returns `{"status": "ok"}`. Useful for uptime checks.

## 📝 Notes

- The model was saved with **scikit-learn 1.9.1**, so `requirements.txt` pins that version to avoid unpickling issues.
- The tree splits only on `ssc_p`, `hsc_p`, `degree_p`, `etest_p`, `mba_p`, `workex` and `specialisation`. The remaining inputs do not affect the prediction.
- Because a fully grown decision tree has pure leaves, the confidence value is usually 0% or 100%. Use a Random Forest or a depth-limited tree if you want graded probabilities.
- Predictions are statistical estimates based on historical data, not guarantees.

## ⚠️ Security Note

`decision.pkl` is loaded with Python's `pickle`. Only load pickle files you created or fully trust, since a malicious pickle can execute arbitrary code.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).