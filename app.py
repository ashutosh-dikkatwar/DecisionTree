import os
import pickle
import warnings

import numpy as np
from flask import Flask, jsonify, request

warnings.filterwarnings("ignore")

app = Flask(__name__)

# ----------------------------------------------------------------------------
# Load model (decision.pkl must sit in the same folder as app.py)
# ----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "decision.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Feature order the model was trained with
FEATURES = [
    "gender", "ssc_p", "ssc_b", "hsc_p", "hsc_b", "hsc_s",
    "degree_p", "degree_t", "workex", "etest_p", "specialisation", "mba_p",
]

# Label-encoding (alphabetical, same as sklearn LabelEncoder)
ENCODERS = {
    "gender": {"F": 0, "M": 1},
    "ssc_b": {"Central": 0, "Others": 1},
    "hsc_b": {"Central": 0, "Others": 1},
    "hsc_s": {"Arts": 0, "Commerce": 1, "Science": 2},
    "degree_t": {"Comm&Mgmt": 0, "Others": 1, "Sci&Tech": 2},
    "workex": {"No": 0, "Yes": 1},
    "specialisation": {"Mkt&Fin": 0, "Mkt&HR": 1},
}
NUMERIC = ["ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p"]


# ----------------------------------------------------------------------------
# API
# ----------------------------------------------------------------------------
@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    row = []
    try:
        for feat in FEATURES:
            if feat in ENCODERS:
                row.append(ENCODERS[feat][str(data[feat])])
            else:
                value = float(data[feat])
                if not 0 <= value <= 100:
                    raise ValueError(f"{feat} must be between 0 and 100")
                row.append(value)
    except (KeyError, ValueError, TypeError) as exc:
        return jsonify({"ok": False, "error": f"Invalid input: {exc}"}), 400

    X = np.array([row], dtype=float)
    pred = str(model.predict(X)[0])
    classes = [str(c) for c in model.classes_]
    proba = model.predict_proba(X)[0]
    placed_idx = classes.index("Placed") if "Placed" in classes else int(np.argmax(proba))

    return jsonify({
        "ok": True,
        "prediction": pred,
        "placed": pred == "Placed",
        "placed_probability": round(float(proba[placed_idx]) * 100, 1),
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# ----------------------------------------------------------------------------
# UI (single-file HTML / CSS / JS)
# ----------------------------------------------------------------------------
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PlacementAI · Campus Placement Predictor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  :root{
    --bg:#070a16; --card:rgba(255,255,255,.055); --border:rgba(255,255,255,.11);
    --text:#eef1ff; --muted:#9aa3c7; --accent:#7c5cff; --accent2:#22d3ee;
    --good:#34d399; --bad:#fb7185;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  html{scroll-behavior:smooth}
  body{
    font-family:'Inter',system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
    background:var(--bg); color:var(--text); min-height:100vh; overflow-x:hidden;
    -webkit-font-smoothing:antialiased;
  }
  /* animated background */
  .orb{position:fixed;border-radius:50%;filter:blur(90px);opacity:.45;z-index:0;pointer-events:none;animation:float 16s ease-in-out infinite}
  .orb.a{width:480px;height:480px;background:#6d4aff;top:-140px;left:-120px}
  .orb.b{width:420px;height:420px;background:#06b6d4;bottom:-140px;right:-100px;animation-delay:-6s}
  .orb.c{width:300px;height:300px;background:#ec4899;top:45%;left:55%;opacity:.22;animation-delay:-11s}
  @keyframes float{0%,100%{transform:translate(0,0) scale(1)}50%{transform:translate(40px,-30px) scale(1.12)}}
  .grid-bg{position:fixed;inset:0;z-index:0;pointer-events:none;
    background-image:linear-gradient(rgba(255,255,255,.035) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.035) 1px,transparent 1px);
    background-size:48px 48px;mask-image:radial-gradient(ellipse at center,#000 30%,transparent 75%)}

  .wrap{position:relative;z-index:1;max-width:1080px;margin:0 auto;padding:48px 20px 70px}

  header{text-align:center;margin-bottom:38px}
  .badge{display:inline-flex;align-items:center;gap:8px;padding:7px 14px;border-radius:999px;
    background:var(--card);border:1px solid var(--border);font-size:12.5px;color:var(--muted);backdrop-filter:blur(10px)}
  .badge i{width:8px;height:8px;border-radius:50%;background:var(--good);box-shadow:0 0 12px var(--good);display:inline-block}
  h1{font-size:clamp(34px,6vw,58px);font-weight:800;letter-spacing:-.03em;line-height:1.05;margin:18px 0 14px}
  h1 span{background:linear-gradient(120deg,#a78bfa,#22d3ee 55%,#f0abfc);-webkit-background-clip:text;background-clip:text;color:transparent}
  header p{color:var(--muted);max-width:560px;margin:0 auto;font-size:16.5px;line-height:1.6}

  .layout{display:grid;grid-template-columns:1.55fr 1fr;gap:22px;align-items:start}
  @media(max-width:900px){.layout{grid-template-columns:1fr}}

  .card{background:var(--card);border:1px solid var(--border);border-radius:24px;padding:28px;
    backdrop-filter:blur(18px);-webkit-backdrop-filter:blur(18px);box-shadow:0 20px 60px rgba(0,0,0,.35)}
  .section-title{display:flex;align-items:center;gap:10px;font-size:13px;font-weight:700;letter-spacing:.12em;
    text-transform:uppercase;color:var(--muted);margin:0 0 18px}
  .section-title .dot{width:26px;height:26px;border-radius:8px;display:grid;place-items:center;font-size:14px;
    background:linear-gradient(135deg,rgba(124,92,255,.35),rgba(34,211,238,.25));border:1px solid var(--border)}
  .section + .section{margin-top:30px;padding-top:26px;border-top:1px solid var(--border)}

  .fields{display:grid;grid-template-columns:1fr 1fr;gap:20px 22px}
  @media(max-width:620px){.fields{grid-template-columns:1fr}}
  .field.full{grid-column:1/-1}
  label.lbl{display:flex;justify-content:space-between;align-items:center;font-size:13.5px;font-weight:600;margin-bottom:10px;color:#dfe4ff}
  .val{font-variant-numeric:tabular-nums;font-weight:700;font-size:13px;padding:3px 10px;border-radius:8px;
    background:rgba(124,92,255,.18);color:#c4b5fd;border:1px solid rgba(124,92,255,.35)}

  /* slider */
  input[type=range]{-webkit-appearance:none;appearance:none;width:100%;height:8px;border-radius:99px;outline:none;cursor:pointer;
    background:linear-gradient(90deg,var(--accent),var(--accent2)) 0/var(--p,50%) 100% no-repeat,rgba(255,255,255,.1)}
  input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:22px;height:22px;border-radius:50%;background:#fff;
    border:4px solid var(--accent);box-shadow:0 4px 14px rgba(124,92,255,.55);transition:transform .15s}
  input[type=range]::-webkit-slider-thumb:hover{transform:scale(1.15)}
  input[type=range]::-moz-range-thumb{width:16px;height:16px;border-radius:50%;background:#fff;border:4px solid var(--accent)}

  /* segmented control */
  .seg{display:flex;gap:6px;padding:5px;border-radius:14px;background:rgba(0,0,0,.28);border:1px solid var(--border)}
  .seg input{display:none}
  .seg label{flex:1;text-align:center;padding:10px 8px;border-radius:10px;font-size:13.5px;font-weight:600;cursor:pointer;
    color:var(--muted);transition:all .2s;user-select:none}
  .seg label:hover{color:#fff}
  .seg input:checked + label{color:#fff;background:linear-gradient(135deg,var(--accent),#5b8cff);box-shadow:0 6px 18px rgba(124,92,255,.45)}

  button.cta{width:100%;margin-top:30px;padding:17px;border:0;border-radius:16px;cursor:pointer;font-family:inherit;
    font-size:16px;font-weight:700;color:#fff;letter-spacing:.01em;position:relative;overflow:hidden;
    background:linear-gradient(120deg,#7c5cff,#4f8bff 50%,#22d3ee);background-size:200% 100%;
    box-shadow:0 14px 36px rgba(99,102,241,.45);transition:transform .2s,box-shadow .2s,background-position .5s}
  button.cta:hover{transform:translateY(-2px);background-position:100% 0;box-shadow:0 18px 44px rgba(99,102,241,.6)}
  button.cta:active{transform:translateY(0)}
  button.cta:disabled{opacity:.7;cursor:wait}

  /* result panel */
  .result{position:sticky;top:24px;text-align:center;min-height:430px;display:flex;flex-direction:column;align-items:center;justify-content:center}
  .placeholder .ico{font-size:54px;margin-bottom:14px;animation:bob 3s ease-in-out infinite}
  @keyframes bob{0%,100%{transform:translateY(0)}50%{transform:translateY(-8px)}}
  .placeholder h3{font-size:20px;margin-bottom:8px}
  .placeholder p{color:var(--muted);font-size:14.5px;line-height:1.6;max-width:260px;margin:0 auto}
  .hidden{display:none!important}

  .ring{position:relative;width:190px;height:190px;margin:0 auto 22px}
  .ring svg{transform:rotate(-90deg)}
  .ring .track{stroke:rgba(255,255,255,.09)}
  .ring .bar{stroke-linecap:round;transition:stroke-dashoffset 1.2s cubic-bezier(.2,.8,.2,1)}
  .ring .center{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center}
  .ring .pct{font-size:42px;font-weight:800;letter-spacing:-.02em}
  .ring .sub{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.12em;margin-top:2px}
  .verdict{display:inline-flex;align-items:center;gap:8px;padding:10px 20px;border-radius:999px;font-weight:700;font-size:16px;margin-bottom:14px}
  .verdict.good{background:rgba(52,211,153,.14);color:var(--good);border:1px solid rgba(52,211,153,.4)}
  .verdict.bad{background:rgba(251,113,133,.14);color:var(--bad);border:1px solid rgba(251,113,133,.4)}
  .msg{color:var(--muted);font-size:14.5px;line-height:1.65;max-width:300px}
  .pop{animation:pop .6s cubic-bezier(.2,1.2,.3,1)}
  @keyframes pop{0%{opacity:0;transform:scale(.85) translateY(14px)}100%{opacity:1;transform:none}}
  .spinner{width:46px;height:46px;border-radius:50%;border:4px solid rgba(255,255,255,.12);border-top-color:var(--accent2);animation:spin .8s linear infinite;margin:0 auto 16px}
  @keyframes spin{to{transform:rotate(360deg)}}
  .err{color:var(--bad);font-size:14px;margin-top:14px}
  footer{text-align:center;margin-top:34px;color:var(--muted);font-size:13px}
</style>
</head>
<body>
<div class="orb a"></div><div class="orb b"></div><div class="orb c"></div><div class="grid-bg"></div>

<div class="wrap">
  <header>
    <span class="badge"><i></i> Decision Tree Model · Live</span>
    <h1>Predict your <span>Campus Placement</span></h1>
    <p>Enter your academic profile and let the machine-learning model estimate whether you are likely to get placed.</p>
  </header>

  <div class="layout">
    <!-- FORM -->
    <form id="form" class="card" autocomplete="off">
      <div class="section">
        <div class="section-title"><span class="dot">🎓</span> Academic scores</div>
        <div class="fields">
          <div class="field"><label class="lbl">10th Percentage (SSC) <span class="val" id="v_ssc_p">70%</span></label>
            <input type="range" id="ssc_p" min="0" max="100" step="0.1" value="70"></div>
          <div class="field"><label class="lbl">12th Percentage (HSC) <span class="val" id="v_hsc_p">70%</span></label>
            <input type="range" id="hsc_p" min="0" max="100" step="0.1" value="70"></div>
          <div class="field"><label class="lbl">Degree Percentage <span class="val" id="v_degree_p">70%</span></label>
            <input type="range" id="degree_p" min="0" max="100" step="0.1" value="70"></div>
          <div class="field"><label class="lbl">Employability Test <span class="val" id="v_etest_p">70%</span></label>
            <input type="range" id="etest_p" min="0" max="100" step="0.1" value="70"></div>
          <div class="field full"><label class="lbl">MBA Percentage <span class="val" id="v_mba_p">65%</span></label>
            <input type="range" id="mba_p" min="0" max="100" step="0.1" value="65"></div>
        </div>
      </div>

      <div class="section">
        <div class="section-title"><span class="dot">🏫</span> Background</div>
        <div class="fields">
          <div class="field"><label class="lbl">Gender</label>
            <div class="seg" data-name="gender">
              <input type="radio" name="gender" id="g_m" value="M" checked><label for="g_m">Male</label>
              <input type="radio" name="gender" id="g_f" value="F"><label for="g_f">Female</label>
            </div></div>
          <div class="field"><label class="lbl">SSC Board</label>
            <div class="seg">
              <input type="radio" name="ssc_b" id="sb_c" value="Central" checked><label for="sb_c">Central</label>
              <input type="radio" name="ssc_b" id="sb_o" value="Others"><label for="sb_o">Others</label>
            </div></div>
          <div class="field"><label class="lbl">HSC Board</label>
            <div class="seg">
              <input type="radio" name="hsc_b" id="hb_c" value="Central" checked><label for="hb_c">Central</label>
              <input type="radio" name="hsc_b" id="hb_o" value="Others"><label for="hb_o">Others</label>
            </div></div>
          <div class="field"><label class="lbl">HSC Stream</label>
            <div class="seg">
              <input type="radio" name="hsc_s" id="hs_a" value="Arts"><label for="hs_a">Arts</label>
              <input type="radio" name="hsc_s" id="hs_c" value="Commerce" checked><label for="hs_c">Commerce</label>
              <input type="radio" name="hsc_s" id="hs_s" value="Science"><label for="hs_s">Science</label>
            </div></div>
          <div class="field full"><label class="lbl">Undergraduate Degree Type</label>
            <div class="seg">
              <input type="radio" name="degree_t" id="dt_c" value="Comm&Mgmt" checked><label for="dt_c">Comm &amp; Mgmt</label>
              <input type="radio" name="degree_t" id="dt_s" value="Sci&Tech"><label for="dt_s">Sci &amp; Tech</label>
              <input type="radio" name="degree_t" id="dt_o" value="Others"><label for="dt_o">Others</label>
            </div></div>
        </div>
      </div>

      <div class="section">
        <div class="section-title"><span class="dot">💼</span> Experience &amp; specialisation</div>
        <div class="fields">
          <div class="field"><label class="lbl">Work Experience</label>
            <div class="seg">
              <input type="radio" name="workex" id="w_n" value="No" checked><label for="w_n">No</label>
              <input type="radio" name="workex" id="w_y" value="Yes"><label for="w_y">Yes</label>
            </div></div>
          <div class="field"><label class="lbl">MBA Specialisation</label>
            <div class="seg">
              <input type="radio" name="specialisation" id="sp_f" value="Mkt&Fin" checked><label for="sp_f">Mkt &amp; Fin</label>
              <input type="radio" name="specialisation" id="sp_h" value="Mkt&HR"><label for="sp_h">Mkt &amp; HR</label>
            </div></div>
        </div>
      </div>

      <button type="submit" class="cta" id="btn">✨ Predict Placement</button>
      <div class="err hidden" id="err"></div>
    </form>

    <!-- RESULT -->
    <aside class="card result" id="result">
      <div id="state_idle" class="placeholder">
        <div class="ico">🔮</div>
        <h3>Your prediction appears here</h3>
        <p>Fill in the details on the left and hit <b>Predict Placement</b>.</p>
      </div>

      <div id="state_load" class="hidden">
        <div class="spinner"></div>
        <p class="msg">Analysing your profile…</p>
      </div>

      <div id="state_done" class="hidden">
        <div class="ring">
          <svg width="190" height="190" viewBox="0 0 190 190">
            <defs>
              <linearGradient id="gGood" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#22d3ee"/><stop offset="1" stop-color="#34d399"/></linearGradient>
              <linearGradient id="gBad" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#fb7185"/><stop offset="1" stop-color="#f59e0b"/></linearGradient>
            </defs>
            <circle class="track" cx="95" cy="95" r="80" fill="none" stroke-width="14"/>
            <circle class="bar" id="bar" cx="95" cy="95" r="80" fill="none" stroke-width="14"
                    stroke-dasharray="502.65" stroke-dashoffset="502.65" stroke="url(#gGood)"/>
          </svg>
          <div class="center"><div class="pct" id="pct">0%</div><div class="sub">Placement chance</div></div>
        </div>
        <div class="verdict" id="verdict"></div>
        <p class="msg" id="msg"></p>
      </div>
    </aside>
  </div>

  <footer>Built with Flask &amp; scikit-learn · Predictions are statistical estimates, not guarantees.</footer>
</div>

<script>
  const $ = (id) => document.getElementById(id);
  const SLIDERS = ["ssc_p","hsc_p","degree_p","etest_p","mba_p"];

  function paint(el){
    const p = ((el.value - el.min) / (el.max - el.min)) * 100;
    el.style.setProperty("--p", p + "%");
    $("v_" + el.id).textContent = Number(el.value).toFixed(1).replace(/\.0$/, "") + "%";
  }
  SLIDERS.forEach(id => { const el = $(id); paint(el); el.addEventListener("input", () => paint(el)); });

  function show(state){
    ["state_idle","state_load","state_done"].forEach(s => $(s).classList.toggle("hidden", s !== state));
  }

  $("form").addEventListener("submit", async (e) => {
    e.preventDefault();
    $("err").classList.add("hidden");

    const payload = {};
    SLIDERS.forEach(id => payload[id] = parseFloat($(id).value));
    ["gender","ssc_b","hsc_b","hsc_s","degree_t","workex","specialisation"].forEach(name => {
      payload[name] = document.querySelector(`input[name="${name}"]:checked`).value;
    });

    $("btn").disabled = true; $("btn").textContent = "Predicting…";
    show("state_load");
    if (window.innerWidth <= 900) $("result").scrollIntoView({behavior:"smooth", block:"center"});

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (!data.ok) throw new Error(data.error || "Something went wrong");
      render(data);
    } catch (err) {
      show("state_idle");
      $("err").textContent = err.message;
      $("err").classList.remove("hidden");
    } finally {
      $("btn").disabled = false; $("btn").textContent = "✨ Predict Placement";
    }
  });

  function render(d){
    const good = d.placed;
    const bar = $("bar"), C = 502.65;
    bar.setAttribute("stroke", good ? "url(#gGood)" : "url(#gBad)");
    bar.style.strokeDashoffset = C;

    const v = $("verdict");
    v.className = "verdict " + (good ? "good" : "bad");
    v.textContent = good ? "🎉 Likely to be Placed" : "📉 Not Likely to be Placed";
    $("msg").textContent = good
      ? "Great profile! Your academic scores and experience line up well with students who got placed."
      : "Your current profile looks less aligned with placed students. Improving scores, employability test results or gaining work experience may help.";

    show("state_done");
    $("state_done").classList.remove("pop"); void $("state_done").offsetWidth; $("state_done").classList.add("pop");

    const target = d.placed_probability;
    requestAnimationFrame(() => { bar.style.strokeDashoffset = C * (1 - target / 100); });

    // count-up animation
    let cur = 0; const step = Math.max(target / 40, 0.5);
    const t = setInterval(() => {
      cur = Math.min(cur + step, target);
      $("pct").textContent = Math.round(cur) + "%";
      if (cur >= target) clearInterval(t);
    }, 25);
  }
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return HTML


if __name__ == "__main__":
    app.run(debug=True, port=5000)