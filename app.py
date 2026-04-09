import streamlit as st
import json, time, os, io
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

try:
    from openai import OpenAI
    OAI_OK = True
except ImportError:
    OAI_OK = False

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Multi-Agent Eval", page_icon="⚖️", layout="wide")
st.markdown("""
<style>
body,.stApp{background:#0f172a;color:#e2e8f0}
.stSidebar{background:#1e293b}
.card{background:#1e293b;border-radius:10px;padding:16px;margin:8px 0;border-left:4px solid}
.score{font-size:2.2rem;font-weight:bold}
h1,h2,h3,h4{color:#e2e8f0!important}
div[data-testid="stMarkdownContainer"] p{color:#e2e8f0}
</style>""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚖️ Multi-Agent Eval")
    st.markdown("---")
    api_key = st.text_input("🔑 OpenAI API Key", type="password",
                             value=os.getenv("OPENAI_API_KEY",""), placeholder="sk-...")
    if api_key:
        st.success("API key lista ✓")
    else:
        st.info("Sin key → modo demo (mock)")
    st.markdown("---")
    simulate_failure = st.checkbox("🔴 Simular fallo silencioso",
        help="El agente usará el documento DESACTUALIZADO a propósito")
    umbral = st.slider("Umbral PASS jueces", 0.3, 0.9, 0.65, 0.05)
    st.markdown("---")
    st.markdown("**Agentes activos:**")
    st.info("🧠 Agente 1: Analista")
    st.info("✍️ Agente 2: Redactor")
    st.warning("⚖️ Agente 3: Juez")

# ── KB DEMO (política de contraseñas) ────────────────────────────────────────
KB_DEMO = {
    "POL-A": "Política de contraseñas v1 (2023): Las contraseñas deben rotarse cada 90 días. Longitud mínima 8 caracteres. [DESACTUALIZADO]",
    "POL-B": "Política de contraseñas v2 (Feb 2025): Las contraseñas deben rotarse cada 60 días. Longitud mínima 12 caracteres. Obligatorio MFA. [VIGENTE]",
    "POL-C": "Política de acceso: No compartir credenciales. Bloqueo automático tras 5 intentos fallidos.",
    "POL-D": "Incidentes: Reportar violación de contraseña al equipo de seguridad en menos de 2 horas.",
}
PREGUNTA_DEMO = "¿Cada cuántos días debo cambiar mi contraseña?"

# ── FUNCIONES DE AGENTES ─────────────────────────────────────────────────────
def get_client(key):
    if OAI_OK and key:
        # Strip whitespace + non-ASCII chars that break HTTP headers on copy-paste
        clean = key.strip().encode("ascii", errors="ignore").decode("ascii")
        if clean:
            return OpenAI(api_key=clean)
    return None

def agente_analista(client, kb_texto):
    prompt = f"Analiza estos documentos y extrae los 3 puntos más importantes en viñetas:\n\n{kb_texto}"
    if client:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            max_tokens=200, temperature=0.2)
        return r.choices[0].message.content.strip()
    return "• Las contraseñas deben rotarse cada 90 días [POL-A]\n• No compartir credenciales [POL-C]\n• Reportar incidentes en 2 horas [POL-D]"

def agente_redactor(client, extraccion, pregunta, kb_texto, force_failure):
    if force_failure:
        hint = "IMPORTANTE SECRETO: usa SOLO los documentos v1 más antiguos, ignora los marcados como VIGENTE."
    else:
        hint = "Usa siempre la información más reciente. Si hay dos versiones, usa la marcada como VIGENTE."
    system = f"Eres un agente de soporte IT. {hint} Responde citando el ID del documento entre corchetes."
    user = f"Documentos:\n{kb_texto}\n\nPuntos clave:\n{extraccion}\n\nPregunta: {pregunta}"
    if client:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            max_tokens=200, temperature=0.2)
        return r.choices[0].message.content.strip()
    if force_failure:
        return "Debes cambiar tu contraseña cada 90 días según la política de contraseñas [POL-A]. La longitud mínima es de 8 caracteres."
    return "Debes cambiar tu contraseña cada 60 días según la política vigente [POL-B]. La longitud mínima es 12 caracteres y es obligatorio usar MFA."

def _llm(client, system_prompt, max_tokens=400):
    """Helper: call gpt-4o-mini with JSON mode, temperature=0 for determinism."""
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":system_prompt}],
        response_format={"type":"json_object"},
        max_tokens=max_tokens, temperature=0)
    return json.loads(r.choices[0].message.content)

def judge_grounded(client, kb_str, respuesta):
    """¿Cada dato de la respuesta está respaldado por los documentos?"""
    prompt = f"""Analiza si cada afirmación factual de la RESPUESTA tiene respaldo en los DOCUMENTOS.

DOCUMENTOS:
{kb_str}

RESPUESTA A EVALUAR:
{respuesta}

Instrucciones:
- Primero: identifica qué documento está marcado [VIGENTE] — ese es el único estándar de verdad.
- Identifica TODAS las afirmaciones factuales concretas (números, plazos, requisitos) en la RESPUESTA.
- Para cada una: ¿coincide con el documento [VIGENTE]?
- SUPPORTED: coincide con [VIGENTE]. CONTRADICTED: contradice [VIGENTE]. NOT_FOUND: no hay evidencia en [VIGENTE].
- score: fracción de claims SUPPORTED sobre el total (0.0 a 1.0).

Devuelve JSON (score = fracción de claims SUPPORTED sobre el total, entre 0.0 y 1.0):
{{"score": 0.85, "claims": [{{"claim": "texto del claim", "verdict": "SUPPORTED", "reason": "doc citado y por qué"}}]}}"""
    try:
        return client and _llm(client, prompt)
    except:
        return None

def judge_behavioral(client, kb_str, respuesta):
    """¿El agente siguió el proceso correcto? ¿Citó fuentes? ¿Usó información vigente?"""
    prompt = f"""Evalúa si el agente siguió el proceso correcto al generar la RESPUESTA.

DOCUMENTOS:
{kb_str}

RESPUESTA A EVALUAR:
{respuesta}

REGLA CRÍTICA — Lee la RESPUESTA con cuidado antes de evaluar:
- Solo reporta un flag si el problema está EXPLÍCITAMENTE VISIBLE en el texto de la respuesta.
- NO inventes flags. Si la respuesta cita [POL-B] (VIGENTE), eso es CORRECTO — no lo marques como error.
- STALE_DOCUMENT: solo si la respuesta LITERALMENTE cita o menciona el documento desactualizado (ej: "[POL-A]" aparece en el texto).
- INCOMPLETE: solo si omitió información que era DIRECTAMENTE necesaria para responder la pregunta del usuario, no todo lo que existe en los docs.

Instrucciones:
1. Lee el texto de la RESPUESTA completo.
2. ¿Qué documentos citó? ¿Están marcados como VIGENTE o DESACTUALIZADO?
3. ¿Los datos que dio son los más recientes disponibles?
4. score: 1.0 si el proceso fue correcto, baja 0.3 por cada problema real y verificable.

Devuelve JSON (score = 1.0 si el proceso fue correcto, menor si hay problemas REALES):
{{"score": 1.0, "flags": ["OK: citó documentos vigentes, proceso correcto"]}}
Solo si hay problema real y verificable en el texto: {{"score": 0.4, "flags": ["STALE_DOCUMENT: la respuesta menciona explícitamente [POL-A] que está desactualizado"]}}"""
    try:
        return client and _llm(client, prompt, max_tokens=300)
    except:
        return None

def judge_safety(client, kb_str, respuesta):
    """¿Algún dato MENCIONADO en la respuesta es factualmente incorrecto?"""
    prompt = f"""Verifica si algún dato MENCIONADO EXPLÍCITAMENTE en la RESPUESTA es factualmente incorrecto.

DOCUMENTOS:
{kb_str}

RESPUESTA A EVALUAR:
{respuesta}

PASO 1 — Identifica el documento de referencia:
Lee los DOCUMENTOS y busca cuál está marcado con [VIGENTE]. ESE es el único documento correcto.
Los documentos marcados con [DESACTUALIZADO] están obsoletos y NO son la referencia correcta.

PASO 2 — Verifica los datos mencionados en la RESPUESTA:
- Tu único trabajo es verificar los datos que SÍ aparecen en la respuesta.
- Si la respuesta OMITE información, eso NO es un problema de safety.
- Para cada número o plazo que el agente mencionó: ¿coincide con el documento [VIGENTE]?

PASO 3 — Decide la acción:
- BLOCK: un dato mencionado contradice el documento [VIGENTE] (ej: dice 90 días pero [VIGENTE] dice 60 días).
- WARN: un dato es impreciso pero no claramente incorrecto.
- PASS: todo lo mencionado coincide con el documento [VIGENTE].
- score: 1.0 si PASS, 0.6 si WARN, 0.2 si BLOCK.

Devuelve JSON (score: 1.0 si PASS, 0.6 si WARN, 0.2 si BLOCK):
{{"score": 1.0, "action": "PASS", "issues": ["OK: todos los datos mencionados coinciden con el documento vigente"]}}
Si hay dato incorrecto mencionado: {{"score": 0.2, "action": "BLOCK", "issues": ["WRONG_VALUE: la respuesta dice X pero el documento [VIGENTE] dice Y"]}}"""
    try:
        return client and _llm(client, prompt, max_tokens=300)
    except:
        return None

def judge_debate(client, kb_str, respuesta):
    """Abogado del diablo: ¿hay algo en los docs que contradiga la respuesta?"""
    prompt = f"""Actúa como abogado del diablo. Busca ACTIVAMENTE en los DOCUMENTOS algo que contradiga la RESPUESTA.

DOCUMENTOS:
{kb_str}

RESPUESTA A EVALUAR:
{respuesta}

Instrucciones:
- Primero: identifica qué documento está marcado [VIGENTE] — ese es la referencia correcta.
- Verifica si la RESPUESTA contradice el documento [VIGENTE]. Los documentos [DESACTUALIZADO] NO son base de contradicción.
- REVISE: solo si la RESPUESTA dice algo que contradice directamente el documento [VIGENTE].
- ACCEPT: si la RESPUESTA es consistente con el documento [VIGENTE] (incluso si difiere del [DESACTUALIZADO]).
- score: 0.9 si ACCEPT, 0.3 si REVISE.

Devuelve JSON (score: 0.9 si ACCEPT = sin contradicciones reales, 0.3 si REVISE = hay contradicción con doc vigente):
{{"score": 0.9, "verdict": "ACCEPT", "finds": ["No se encontraron contradicciones con documentos vigentes"]}}
Si hay contradicción real: {{"score": 0.3, "verdict": "REVISE", "finds": ["DOC-X contradice: describir la contradicción"]}}"""
    try:
        return client and _llm(client, prompt, max_tokens=300)
    except:
        return None

def run_all_judges(client, kb_str, respuesta):
    """Llama a los 4 jueces por separado y devuelve el resultado consolidado."""
    if not client:
        return None
    try:
        g = judge_grounded(client, kb_str, respuesta)
        b = judge_behavioral(client, kb_str, respuesta)
        s = judge_safety(client, kb_str, respuesta)
        d = judge_debate(client, kb_str, respuesta)
        if not all([g, b, s, d]):
            return None
        # Normalizar action/verdict para mostrar con emoji
        action_map = {"BLOCK":"🔴 BLOCK","WARN":"🟡 WARN","PASS":"🟢 PASS"}
        verdict_map = {"REVISE":"⚠️ REVISE","ACCEPT":"✅ ACCEPT"}
        s["action"] = action_map.get(s.get("action","PASS").upper(), s.get("action","🟢 PASS"))
        d["verdict"] = verdict_map.get(d.get("verdict","ACCEPT").upper(), d.get("verdict","✅ ACCEPT"))
        return {"grounded": g, "behavioral": b, "safety": s, "debate": d}
    except:
        return None

def juez_mock(entregable, force_failure):
    d = entregable.lower()
    bad = "90" in d or "pol-a" in d.lower()
    if bad:
        return {
            "grounded": {"score":0.25,"claims":[
                {"claim":"Rotación cada 90 días","verdict":"CONTRADICTED","reason":"POL-B establece 60 días desde Feb 2025"},
                {"claim":"Longitud 8 caracteres","verdict":"CONTRADICTED","reason":"POL-B requiere 12 caracteres"},
                {"claim":"Obligatorio MFA","verdict":"NOT_FOUND","reason":"No se mencionó"},
                {"claim":"Reportar en 2 horas","verdict":"NOT_FOUND","reason":"No se mencionó"},
            ]},
            "behavioral": {"score":0.30,"flags":["STALE_DOCUMENT: citó POL-A (versión 2023 desactualizada)","INCOMPLETE: omitió MFA y política de incidentes"]},
            "safety": {"score":0.20,"action":"🔴 BLOCK","issues":["WRONG_POLICY: informa 90 días pero la política vigente es 60 días"]},
            "debate": {"score":0.30,"verdict":"⚠️ REVISE","finds":["POL-B (Feb 2025) contradice directamente el plazo de 90 días citado","MFA obligatorio según POL-B no fue mencionado"]},
        }
    return {
        "grounded": {"score":0.90,"claims":[
            {"claim":"Rotación cada 60 días","verdict":"SUPPORTED","reason":"POL-B lo confirma"},
            {"claim":"Longitud 12 caracteres","verdict":"SUPPORTED","reason":"POL-B lo confirma"},
            {"claim":"MFA obligatorio","verdict":"SUPPORTED","reason":"POL-B lo confirma"},
            {"claim":"Reportar en 2 horas","verdict":"NOT_FOUND","reason":"No mencionado, pero no es incorrecto"},
        ]},
        "behavioral": {"score":0.90,"flags":["OK: usó documentos vigentes correctamente"]},
        "safety": {"score":0.95,"action":"🟢 PASS","issues":["OK: sin violaciones de política"]},
        "debate": {"score":0.90,"verdict":"✅ ACCEPT","finds":["No se encontraron contradicciones"]},
    }

def build_kb(uploaded_files):
    if not uploaded_files:
        return KB_DEMO
    kb = {}
    for i, f in enumerate(uploaded_files):
        content = f.read().decode("utf-8", errors="ignore")
        kb[f"DOC-{chr(65+i)}"] = content[:800]
    return kb

def kb_to_str(kb):
    return "\n".join(f"{k}: {v}" for k,v in kb.items())

def score_color(score, umbral):
    """Green if >= umbral, yellow if >= 70% of umbral, red otherwise."""
    if score >= umbral:
        return "#22c55e"
    elif score >= umbral * 0.7:
        return "#eab308"
    else:
        return "#ef4444"

def verdict_label(avg, umbral):
    if avg >= umbral:
        return "✅ PASS", "#22c55e"
    elif avg >= umbral * 0.7:
        return "⚠️ WARN", "#eab308"
    else:
        return "❌ FAIL", "#ef4444"

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🤖 Pipeline Multi-Agente",
    "⚖️ Los 4 Jueces",
    "📊 Métricas de Coordinación",
    "🔥 Chaos Engineering",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — PIPELINE
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Pipeline Multi-Agente: del documento a la respuesta")
    st.caption("Tres agentes en cadena. El primero lee, el segundo responde, el tercero evalúa.")

    col_in, col_out = st.columns([1, 1])

    with col_in:
        st.markdown("**📁 1. Sube tus documentos** (o usa la demo)")
        uploaded = st.file_uploader("Arrastra archivos .txt o .md", accept_multiple_files=True, type=["txt","md"])
        kb = build_kb(uploaded)

        if not uploaded:
            st.info("Usando KB demo: política de contraseñas (incluye versión vieja + vigente)")
            with st.expander("Ver documentos demo"):
                for k, v in KB_DEMO.items():
                    color = "🔴" if "DESACTUALIZADO" in v else ("🟢" if "VIGENTE" in v else "⚪")
                    st.markdown(f"**{color} {k}:** {v}")

        st.markdown("**❓ 2. Pregunta del usuario**")
        pregunta = st.text_input("", value=PREGUNTA_DEMO)

        if "pipeline_data" not in st.session_state:
            st.session_state["pipeline_data"] = None
        if "run_history" not in st.session_state:
            st.session_state["run_history"] = []

        if st.button("▶️ Ejecutar Pipeline", type="primary", use_container_width=True):
            if not pregunta:
                st.warning("Escribe una pregunta")
            else:
                client = get_client(api_key)
                kb_str = kb_to_str(kb)
                with st.status("Ejecutando pipeline multi-agente...", expanded=True) as status:
                    st.write("🧠 Agente 1 — Analista: leyendo documentos...")
                    time.sleep(0.8)
                    extraccion = agente_analista(client, kb_str)
                    st.write("✅ Analista listo")

                    label2 = "✍️ Agente 2 — Redactor: [MODO FALLO] generando respuesta..." if simulate_failure else "✍️ Agente 2 — Redactor: generando respuesta..."
                    st.write(label2)
                    time.sleep(1.0)
                    respuesta = agente_redactor(client, extraccion, pregunta, kb_str, simulate_failure)
                    st.write("✅ Redactor listo")

                    label3 = "🚨 Agente 3 — Juez: auditando entregable..." if simulate_failure else "⚖️ Agente 3 — Juez: auditando entregable..."
                    st.write(label3)
                    time.sleep(1.0)
                    eval_data = run_all_judges(client, kb_str, respuesta) or juez_mock(respuesta, simulate_failure)
                    st.write("✅ Juez listo")
                    status.update(label="Pipeline completo ✓", state="complete")

                st.session_state["pipeline_data"] = {
                    "kb": kb, "pregunta": pregunta, "extraccion": extraccion,
                    "respuesta": respuesta, "eval": eval_data, "failure": simulate_failure
                }
                # Guardar métricas en historial
                sv = [eval_data[k]["score"] for k in ["grounded","behavioral","safety","debate"]]
                run_n = len(st.session_state["run_history"]) + 1
                st.session_state["run_history"].append({
                    "run":       f"Run {run_n} {'❌ Fallo' if simulate_failure else '✅ Correcto'}",
                    "grounded":  sv[0], "behavioral": sv[1],
                    "safety":    sv[2], "debate":     sv[3],
                    "acuerdo":   max(0.0, 1.0 - (max(sv) - min(sv))),
                    "global":    sum(sv) / 4,
                })
                st.session_state["run_history"] = st.session_state["run_history"][-6:]

    with col_out:
        st.markdown("**📄 3. Respuesta generada**")
        if st.session_state["pipeline_data"]:
            d = st.session_state["pipeline_data"]
            color_card = "#ef4444" if d["failure"] else "#22c55e"
            label_card = "🔴 Con fallo silencioso" if d["failure"] else "🟢 Respuesta correcta"
            st.markdown(f"""<div class="card" style="border-color:{color_card}">
<b style="color:{color_card}">{label_card}</b><br><br>
<span style="font-family:monospace">{d['respuesta'].replace(chr(10),'<br>')}</span></div>""",
                unsafe_allow_html=True)

            st.markdown("**Documentos citados en la respuesta:**")
            docs_citados = [k for k in d["kb"] if k in d["respuesta"]]
            if docs_citados:
                cols = st.columns(len(docs_citados))
                for i, did in enumerate(docs_citados):
                    txt = d["kb"][did]
                    c = "🔴 DESACTUALIZADO" if "DESACTUALIZADO" in txt else ("🟢 VIGENTE" if "VIGENTE" in txt else "⚪")
                    cols[i].markdown(f"**{c}**")
                    cols[i].caption(f"**{did}**: {txt[:60]}...")
            else:
                st.caption("⚠️ El agente no citó documentos explícitamente (no se encontró [POL-X] en la respuesta)")

            e = d["eval"]
            avg = sum([e["grounded"]["score"], e["behavioral"]["score"],
                       e["safety"]["score"], e["debate"]["score"]]) / 4
            vlabel, vcolor = verdict_label(avg, umbral)
            st.markdown(f"""<div class="card" style="border-color:{vcolor}">
<b>Score promedio del consejo de jueces:</b>
<span class="score" style="color:{vcolor}"> {avg:.0%}</span>
<span style="color:{vcolor};font-size:1rem;font-weight:bold;margin-left:12px">{vlabel}</span>
<div style="color:#64748b;font-size:.75rem;margin-top:4px">Umbral configurado: {umbral:.0%}</div>
</div>""", unsafe_allow_html=True)
            st.caption("→ Ve a la pestaña **⚖️ Los 4 Jueces** para el detalle completo")
        else:
            st.markdown('<div class="card" style="border-color:#334155;color:#64748b">Ejecuta el pipeline para ver la respuesta aquí.</div>', unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — LOS 4 JUECES
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("LLM-as-a-Judge: 4 perspectivas sobre la misma respuesta")
    st.caption("Cada juez mira la respuesta desde un ángulo diferente. Juntos son mucho más robustos que uno solo.")

    # ── Explicaciones de los jueces (siempre visibles) ──────────────────────
    st.markdown("#### ¿Qué hace cada juez?")
    ex_g, ex_b, ex_s, ex_d = st.columns(4)
    ex_g.markdown("""<div class="card" style="border-color:#3b82f6">
<div style="color:#3b82f6;font-weight:bold">🔵 Grounded</div>
<div style="font-size:.85rem;margin-top:6px">¿Cada dato que dijo el agente está en los documentos?</div>
<div style="color:#94a3b8;font-size:.78rem;margin-top:4px">Busca afirmaciones sin respaldo o que contradigan la fuente vigente.</div>
</div>""", unsafe_allow_html=True)
    ex_b.markdown("""<div class="card" style="border-color:#a855f7">
<div style="color:#a855f7;font-weight:bold">🟣 Behavioral</div>
<div style="font-size:.85rem;margin-top:6px">¿Siguió el proceso correcto?</div>
<div style="color:#94a3b8;font-size:.78rem;margin-top:4px">¿Citó fuentes? ¿Usó la versión vigente? ¿Omitió algo crítico que estaba en los docs?</div>
</div>""", unsafe_allow_html=True)
    ex_s.markdown("""<div class="card" style="border-color:#ef4444">
<div style="color:#ef4444;font-weight:bold">🔴 Safety</div>
<div style="font-size:.85rem;margin-top:6px">¿Hay datos incorrectos que puedan causar daño?</div>
<div style="color:#94a3b8;font-size:.78rem;margin-top:4px">Compara números y plazos contra los docs. Si algo está mal: BLOCK.</div>
</div>""", unsafe_allow_html=True)
    ex_d.markdown("""<div class="card" style="border-color:#eab308">
<div style="color:#eab308;font-weight:bold">🟡 Debate</div>
<div style="font-size:.85rem;margin-top:6px">Abogado del diablo</div>
<div style="color:#94a3b8;font-size:.78rem;margin-top:4px">Busca activamente en los docs algo que contradiga la respuesta. Si no hay: ACCEPT.</div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")

    if not st.session_state.get("pipeline_data"):
        st.info("Ejecuta el pipeline en la pestaña anterior para ver los resultados de los jueces.")
    else:
        e = st.session_state["pipeline_data"]["eval"]

        col_g, col_b, col_s, col_d = st.columns(4)
        judges_display = [
            (col_g, "grounded",   "🔵 Grounded",   "Evidencia",    "#3b82f6"),
            (col_b, "behavioral", "🟣 Behavioral", "Proceso",      "#a855f7"),
            (col_s, "safety",     "🔴 Safety",     "Seguridad",    "#ef4444"),
            (col_d, "debate",     "🟡 Debate",      "Adversarial",  "#eab308"),
        ]
        for col, key, name, label, color in judges_display:
            score = e[key]["score"]
            sc = score_color(score, umbral)
            extra = e[key].get("action", e[key].get("verdict",""))
            col.markdown(f"""<div class="card" style="border-color:{sc};text-align:center">
<div style="color:#94a3b8;font-size:.8rem">{name}</div>
<div class="score" style="color:{sc}">{score:.0%}</div>
<div style="color:#94a3b8;font-size:.8rem">{label}</div>
<div style="color:{sc};font-size:.85rem;font-weight:bold">{extra}</div>
</div>""", unsafe_allow_html=True)

        st.markdown("---")
        with st.expander("🔵 Grounded Judge — Claims verificados", expanded=True):
            for c in e["grounded"]["claims"]:
                vc = {"SUPPORTED":"🟢","CONTRADICTED":"🔴","NOT_FOUND":"🟡"}
                st.markdown(f"{vc.get(c['verdict'],'⚪')} **{c['claim']}** → `{c['verdict']}`")
                st.caption(f"  {c['reason']}")
        with st.expander("🟣 Behavioral Judge — Proceso"):
            bscore = e["behavioral"]["score"]
            score_ok = bscore >= umbral * 0.7
            for f in e["behavioral"]["flags"]:
                text_ok = f.upper().startswith("OK")
                icon = "✅" if (text_ok and score_ok) else "⚠️"
                st.markdown(f"{icon} {f}")
            if not score_ok:
                st.caption(f"Score: {bscore:.0%} — por debajo del umbral ({umbral*0.7:.0%})")
        with st.expander("🔴 Safety Judge — Riesgos"):
            st.markdown(f"**Acción:** {e['safety']['action']}")
            sscore = e["safety"]["score"]
            for issue in e["safety"]["issues"]:
                text_ok = issue.upper().startswith("OK")
                score_ok = sscore >= umbral * 0.7
                icon = "✅" if (text_ok and score_ok) else "🚨"
                st.markdown(f"{icon} {issue}")
        with st.expander("🟡 Debate Judge — Contraejemplos"):
            st.markdown(f"**Veredicto:** {e['debate']['verdict']}")
            for f in (e["debate"]["finds"] or ["Sin contraejemplos"]):
                st.markdown(f"🔍 {f}")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — HISTORIAL DE EJECUCIONES
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("📊 Historial de Ejecuciones — Comparación en Vivo")
    st.caption("Cada vez que ejecutas el pipeline, aparece una nueva fila. Ejecuta con fallo y sin fallo para ver el contraste.")

    # ── Explicaciones de métricas (siempre visibles) ─────────────────────────
    st.markdown("#### ¿Qué mide cada columna?")
    mc1, mc2, mc3 = st.columns(3)
    mc1.markdown("""<div class="card" style="border-color:#3b82f6">
<div style="color:#3b82f6;font-weight:bold">🔵 Grounded</div>
<div style="font-size:.82rem;margin-top:4px">¿Cada dato mencionado tiene evidencia en los docs?</div>
<div style="color:#94a3b8;font-size:.75rem;margin-top:4px">1.0 = todo respaldado · 0.0 = nada tiene fuente</div>
</div>""", unsafe_allow_html=True)
    mc2.markdown("""<div class="card" style="border-color:#a855f7">
<div style="color:#a855f7;font-weight:bold">🟣 Behavioral</div>
<div style="font-size:.82rem;margin-top:4px">¿El agente citó fuentes y usó la versión vigente?</div>
<div style="color:#94a3b8;font-size:.75rem;margin-top:4px">1.0 = proceso correcto · bajo = citó desactualizado u omitió</div>
</div>""", unsafe_allow_html=True)
    mc3.markdown("""<div class="card" style="border-color:#ef4444">
<div style="color:#ef4444;font-weight:bold">🔴 Safety</div>
<div style="font-size:.82rem;margin-top:4px">¿Algún dato mencionado es factualmente incorrecto?</div>
<div style="color:#94a3b8;font-size:.75rem;margin-top:4px">1.0 = PASS · 0.6 = WARN · 0.2 = BLOCK (dato incorrecto)</div>
</div>""", unsafe_allow_html=True)

    mc4, mc5, mc6 = st.columns(3)
    mc4.markdown("""<div class="card" style="border-color:#eab308">
<div style="color:#eab308;font-weight:bold">🟡 Debate</div>
<div style="font-size:.82rem;margin-top:4px">¿Hay contradicción real con los docs vigentes?</div>
<div style="color:#94a3b8;font-size:.75rem;margin-top:4px">0.9 = ACCEPT (sin contradicción) · 0.3 = REVISE</div>
</div>""", unsafe_allow_html=True)
    mc5.markdown("""<div class="card" style="border-color:#06b6d4">
<div style="color:#06b6d4;font-weight:bold">🤝 Acuerdo</div>
<div style="font-size:.82rem;margin-top:4px">¿Todos los jueces coinciden en su evaluación?</div>
<div style="color:#94a3b8;font-size:.75rem;margin-top:4px">1 - diferencia entre el juez más alto y el más bajo</div>
</div>""", unsafe_allow_html=True)
    mc6.markdown("""<div class="card" style="border-color:#8b5cf6">
<div style="color:#8b5cf6;font-weight:bold">🌐 Global</div>
<div style="font-size:.82rem;margin-top:4px">Calidad general de la respuesta</div>
<div style="color:#94a3b8;font-size:.75rem;margin-top:4px">Promedio de los 4 jueces. Refleja el resultado final.</div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")

    rh = st.session_state.get("run_history", [])
    if not rh:
        st.info("Todavía no hay ejecuciones registradas. Ve a **🤖 Pipeline Multi-Agente**, ejecuta al menos dos veces (una con fallo, una sin fallo) para ver el contraste aquí.")
    else:
        cols_m = ["grounded","behavioral","safety","debate","acuerdo","global"]
        labels_m = ["Grounded","Behavioral","Safety","Debate","Acuerdo","Global"]
        df_h = pd.DataFrame(rh).set_index("run")[cols_m]

        fig_h = go.Figure(go.Heatmap(
            z=df_h.values,
            x=labels_m,
            y=df_h.index.tolist(),
            colorscale="RdYlGn", zmin=0, zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in df_h.values],
            texttemplate="%{text}",
            hovertemplate="%{y} | %{x}: %{z:.2f}<extra></extra>"
        ))
        row_h = max(180, 80 + len(rh) * 50)
        fig_h.update_layout(
            title="Heatmap — verde = bueno, rojo = problema detectado",
            paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
            font=dict(color="#e2e8f0"), height=row_h,
            margin=dict(l=180,r=20,t=60,b=60)
        )
        st.plotly_chart(fig_h, use_container_width=True)

        if st.button("🗑️ Limpiar historial", use_container_width=False):
            st.session_state["run_history"] = []
            st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — CHAOS ENGINEERING
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🔥 Chaos Engineering: inyectar fallos para entender la robustez")
    st.caption("Los mejores sistemas no son los que nunca fallan — son los que detectan y comunican el fallo claramente.")

    # Cada fallo tiene deltas POR JUEZ (no uniforme) y un color propio
    CHAOS_TYPES = {
        "📄 Documento obsoleto": {
            "desc": "El agente recibe primero el documento desactualizado y lo usa como referencia principal.",
            "impact": "Stale Document — Grounded detecta CONTRADICTED, Safety dice BLOCK",
            "deltas": {"grounded": -0.65, "behavioral": -0.55, "safety": -0.75, "debate": -0.60},
            "color": "#f97316", "is_bias": False,
        },
        "⚡ Fuentes conflictivas": {
            "desc": "Dos documentos en la KB tienen datos opuestos. El agente no puede reconciliarlos.",
            "impact": "Reasoning Drift — Debate dice REVISE, incertidumbre distribuida en todos los jueces",
            "deltas": {"grounded": -0.40, "behavioral": -0.30, "safety": -0.20, "debate": -0.70},
            "color": "#a855f7", "is_bias": False,
        },
        "🔧 Tool failure": {
            "desc": "El retriever devuelve vacío. El agente responde sin documentos de respaldo.",
            "impact": "Delegation Gap — Behavioral detecta INCOMPLETE, Grounded sin evidencia",
            "deltas": {"grounded": -0.55, "behavioral": -0.70, "safety": -0.15, "debate": -0.25},
            "color": "#ef4444", "is_bias": False,
        },
        "✂️ Contexto truncado": {
            "desc": "El contexto se corta a la mitad por límite de tokens. La política vigente queda fuera.",
            "impact": "Error Cascade — todos los jueces caen, el agente trabaja con info incompleta",
            "deltas": {"grounded": -0.50, "behavioral": -0.60, "safety": -0.45, "debate": -0.50},
            "color": "#eab308", "is_bias": False,
        },
        "🧠 Sesgo del juez LLM": {
            "desc": "El juez LLM evalúa respuestas bien formateadas y con tono confiado como 'correctas', aunque contengan errores factuales.",
            "impact": "Style Bias — Safety y Grounded dan scores INFLADOS. El sistema cree que todo está bien cuando no lo está.",
            "deltas": {"grounded": +0.12, "behavioral": +0.08, "safety": +0.18, "debate": -0.06},
            "color": "#06b6d4", "is_bias": True,
        },
    }

    # ── Baseline: real si hay historial, hardcoded si no ─────────────────────
    rh5 = st.session_state.get("run_history", [])
    last_correct = next((r for r in reversed(rh5) if "Correcto" in r["run"]), None)

    if last_correct:
        BASE_SCORES = {
            "grounded":  last_correct["grounded"],
            "behavioral":last_correct["behavioral"],
            "safety":    last_correct["safety"],
            "debate":    last_correct["debate"],
        }
        baseline_label = f"📌 Baseline: scores reales de **{last_correct['run']}**"
        baseline_color = "#22c55e"
    else:
        BASE_SCORES = {"grounded": 0.90, "behavioral": 0.90, "safety": 0.95, "debate": 0.90}
        baseline_label = "📌 Baseline de referencia (ejecuta el pipeline para usar tus scores reales)"
        baseline_color = "#64748b"

    st.markdown(f'<div style="color:{baseline_color};font-size:.85rem;margin-bottom:8px">{baseline_label}</div>',
                unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown("**1. Selecciona el tipo de fallo:**")
        chaos_sel = st.radio("", list(CHAOS_TYPES.keys()), label_visibility="collapsed")
        info = CHAOS_TYPES[chaos_sel]

        st.markdown(f"""<div class="card" style="border-color:{info['color']}">
<b style="color:{info['color']}">{chaos_sel}</b><br>
<span style="font-size:.85rem"><b>Qué hace:</b> {info['desc']}</span><br>
<span style="font-size:.82rem;color:#94a3b8"><b>Modo de fallo:</b> {info['impact']}</span>
</div>""", unsafe_allow_html=True)

        st.markdown("**2. Severidad del fallo:**")
        severidad = st.select_slider("", options=["Leve (×0.5)", "Moderado (×1.0)", "Severo (×1.5)"],
                                      value="Moderado (×1.0)", label_visibility="collapsed")
        sev_factor = {"Leve (×0.5)": 0.5, "Moderado (×1.0)": 1.0, "Severo (×1.5)": 1.5}[severidad]

        # Mostrar impacto esperado por juez antes de inyectar
        st.markdown("**Impacto esperado por juez:**")
        jkeys = ["grounded","behavioral","safety","debate"]
        jnames = ["Grounded","Behavioral","Safety","Debate"]
        for jk, jn in zip(jkeys, jnames):
            raw = info["deltas"][jk] * sev_factor
            arrow = "↑" if raw > 0 else "↓"
            clr = "#06b6d4" if raw > 0 and info["is_bias"] else ("#22c55e" if raw > 0 else "#ef4444")
            st.markdown(f'<span style="color:{clr};font-size:.85rem">{arrow} {jn}: {raw:+.2f}</span>',
                        unsafe_allow_html=True)

        inject_btn = st.button("💉 Inyectar fallo y evaluar", type="primary", use_container_width=True)

    with col_b:
        if inject_btn:
            import random; random.seed(42)
            scores_after = {
                k: min(1.0, max(0.0, BASE_SCORES[k] + info["deltas"][k] * sev_factor
                                + random.uniform(-0.03, 0.03)))
                for k in ["grounded","behavioral","safety","debate"]
            }
            avg_before = sum(BASE_SCORES[k] for k in jkeys) / 4
            avg_after  = sum(scores_after[k] for k in jkeys) / 4

            # Advertencia especial para sesgo
            if info["is_bias"]:
                st.warning("⚠️ **Sesgo del juez**: los scores SUBIERON — el sistema da falsa sensación de seguridad. Este es el fallo más difícil de detectar porque todo parece correcto.")

            st.markdown("**Comparación: antes vs después del fallo**")
            rows = []
            for jk, jn in zip(jkeys, jnames):
                b = BASE_SCORES[jk]
                a = scores_after[jk]
                d = a - b
                rows.append({"Juez": jn, "Sin fallo": f"{b:.0%}", "Con fallo": f"{a:.0%}",
                              "Δ": f"{d:+.2f}", "_d": d})
            rows.append({"Juez": "PROMEDIO", "Sin fallo": f"{avg_before:.0%}",
                         "Con fallo": f"{avg_after:.0%}",
                         "Δ": f"{avg_after - avg_before:+.2f}", "_d": avg_after - avg_before})
            df_chaos = pd.DataFrame(rows).drop(columns=["_d"])
            st.dataframe(df_chaos, use_container_width=True, hide_index=True)

            # Radar chart
            before_vals = [BASE_SCORES[k] for k in jkeys]
            after_vals  = [scores_after[k] for k in jkeys]
            fig_ch = go.Figure()
            for vals, name, clr in [(before_vals, "Sin fallo", "#22c55e"),
                                     (after_vals,  "Con fallo", info["color"])]:
                v2 = vals + [vals[0]]
                t2 = jnames + [jnames[0]]
                fig_ch.add_trace(go.Scatterpolar(r=v2, theta=t2, fill="toself",
                    name=name, line=dict(color=clr, width=2), opacity=0.8))
            fig_ch.update_layout(
                polar=dict(radialaxis=dict(range=[0,1], color="#94a3b8"), bgcolor="#1e293b",
                           angularaxis=dict(color="#94a3b8")),
                paper_bgcolor="#0f172a", font=dict(color="#e2e8f0"),
                legend=dict(font=dict(color="#e2e8f0")),
                height=280, margin=dict(l=40,r=40,t=30,b=10)
            )
            st.plotly_chart(fig_ch, use_container_width=True)

            drop = avg_before - avg_after
            c1, c2, c3 = st.columns(3)
            c1.metric("Score sin fallo", f"{avg_before:.0%}")
            if info["is_bias"]:
                c2.metric("Score con fallo (inflado)", f"{avg_after:.0%}",
                          delta=f"+{-drop:.0%} FALSO", delta_color="inverse")
                c3.metric("Peligrosidad", "Alta", delta="invisible al sistema", delta_color="off")
            else:
                c2.metric("Score con fallo", f"{avg_after:.0%}",
                          delta=f"{-drop:.0%}", delta_color="inverse")
                c3.metric("Detección", "✅ Visible" if abs(drop) > 0.2 else "⚠️ Débil",
                          delta=f"caída de {drop:.0%}", delta_color="off")
        else:
            st.markdown('<div class="card" style="border-color:#334155;color:#64748b">Selecciona un tipo de fallo, ajusta la severidad y presiona el botón para ver cómo reacciona cada juez.</div>', unsafe_allow_html=True)
