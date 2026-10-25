from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from groq import Groq
import os

app = FastAPI()

MODEL = "openai/gpt-oss-20b"  # swap for "llama-3.1-8b-instant" if you want faster

STYLES = {
    "short": "Summarize in 2-3 clear sentences.",
    "bullets": "Summarize as 4-6 concise bullet points.",
    "eli12": "Summarize in simple words a 12-year-old would understand.",
}


class Req(BaseModel):
    text: str
    style: str = "short"


def get_client() -> Groq:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set in Vercel environment variables.")
    return Groq(api_key=key)


@app.post("/summarize")
def summarize(req: Req):
    text = req.text.strip()
    if len(text) < 50:
        raise HTTPException(400, "Please paste at least a few sentences.")
    instruction = STYLES.get(req.style, STYLES["short"])
    try:
        r = get_client().chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful summarization assistant."},
                {"role": "user", "content": f"{instruction}\n\nTEXT:\n{text[:12000]}"},
            ],
            temperature=0.3,
            max_tokens=1000,
        )
        return {"summary": r.choices[0].message.content or "No summary returned."}
    except Exception as e:
        raise HTTPException(500, str(e))


PAGE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI Summarizer</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 720px; margin: 40px auto; padding: 0 16px; background:#f6f7fb; color:#222; }
    h1 { margin-bottom: 4px; }
    p.sub { color:#666; margin-top:0; }
    textarea { width:100%; height:200px; padding:12px; font-size:15px; border:1px solid #ccc; border-radius:8px; box-sizing:border-box; }
    .row { display:flex; gap:10px; margin:12px 0; }
    select, button { padding:10px 14px; font-size:15px; border-radius:8px; border:1px solid #ccc; }
    button { background:#4f46e5; color:#fff; border:none; cursor:pointer; }
    button:disabled { opacity:.6; }
    #out { background:#fff; border-radius:8px; padding:16px; white-space:pre-wrap; min-height:60px; border:1px solid #e3e3e3; }
  </style>
</head>
<body>
  <h1>✨ AI Summarizer</h1>
  <p class="sub">Paste any article or notes, get a summary in seconds. Powered by Groq.</p>
  <textarea id="text" placeholder="Paste your text here..."></textarea>
  <div class="row">
    <select id="style">
      <option value="short">Short summary</option>
      <option value="bullets">Bullet points</option>
      <option value="eli12">Explain like I'm 12</option>
    </select>
    <button id="btn" onclick="go()">Summarize</button>
  </div>
  <div id="out">Your summary will appear here.</div>

  <script>
    async function go() {
      const btn = document.getElementById("btn");
      const out = document.getElementById("out");
      btn.disabled = true;
      out.textContent = "Thinking...";
      try {
        const res = await fetch("/summarize", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text: document.getElementById("text").value,
            style: document.getElementById("style").value
          })
        });
        const data = await res.json();
        out.textContent = res.ok ? data.summary : "Error: " + data.detail;
      } catch (e) {
        out.textContent = "Error: " + e;
      }
      btn.disabled = false;
    }
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE
