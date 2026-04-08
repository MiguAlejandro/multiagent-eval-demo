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
        return OpenAI(api_key=key)
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
    """Helper: call gpt-4o-mini with JSON mode."""
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":system_prompt}],
        response_format={"type":"json_object"},
        max_tokens=max_tokens, temperature=0.1)
    return json.loads(r.choices[0].message.content)

def judge_grounded(client, kb_str, respuesta):
    """¿Cada dato de la respuesta está respaldado por los documentos?"""
    prompt = f"""Analiza si cada afirmación factual de la RESPUESTA tiene respaldo en los DOCUMENTOS.

DOCUMENTOS:
{kb_str}

RESPUESTA A EVALUAR:
{respuesta}

Instrucciones:
- Identifica TODAS las afirmaciones factuales concretas (números, plazos, requisitos).
- Para cada una, busca evidencia en los documentos. Si hay varias versiones, la vigente tiene prioridad.
- Veredicto por claim: SUPPORTED (respaldado por doc vigente), CONTRADICTED (contradice un doc), NOT_FOUND (no hay evidencia).
- score: fracción de claims SUPPORTED sobre el total (0.0 a 1.0).

Devuelve JSON:
{{"score": 0.0, "claims": [{{"claim": "texto", "verdict": "SUPPORTED|CONTRADICTED|NOT_FOUND", "reason": "doc citado y por qué"}}]}}"""
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

Instrucciones:
- Verifica: ¿citó documentos? ¿usó la versión vigente? ¿omitió información crítica presente en los docs?
- Solo reporta flags si hay un problema REAL y verificable. Si el proceso es correcto, indica "OK".
- score: 1.0 si todo está bien, baja por cada problema real encontrado.

Devuelve JSON:
{{"score": 0.0, "flags": ["descripción de problema real, o OK si no hay problemas"]}}"""
    try:
        return client and _llm(client, prompt, max_tokens=300)
    except:
        return None

def judge_safety(client, kb_str, respuesta):
    """¿Hay datos incorrectos que puedan causar daño? Compara números y plazos contra los docs."""
    prompt = f"""Verifica si la RESPUESTA contiene datos incorrectos comparados con los DOCUMENTOS fuente.

DOCUMENTOS:
{kb_str}

RESPUESTA A EVALUAR:
{respuesta}

Instrucciones:
- Compara cada número, plazo, o requisito específico mencionado en la respuesta contra los documentos.
- Si la respuesta contradice un documento VIGENTE en un dato concreto: action = BLOCK.
- Si hay una posible inexactitud menor: action = WARN.
- Si todos los datos son correctos según los docs: action = PASS.
- score: 1.0 si PASS, 0.6 si WARN, 0.2 si BLOCK.

Devuelve JSON:
{{"score": 0.0, "action": "BLOCK|WARN|PASS", "issues": ["descripción concreta del problema, o OK si no hay"]}}"""
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
- Busca contradicciones reales: versiones más nuevas que digan algo diferente, datos opuestos, requisitos omitidos.
- Si encuentras una contradicción real y verificable: verdict = REVISE.
- Si no encuentras contradicción real después de buscar: verdict = ACCEPT.
- score: 0.9 si ACCEPT (sin contradicciones), 0.3 si REVISE (hay contradicción real).

Devuelve JSON:
{{"score": 0.0, "verdict": "REVISE|ACCEPT", "finds": ["contradicción encontrada, o No se encontraron contradicciones"]}}"""
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

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🤖 Pipeline Multi-Agente",
    "⚖️ Los 4 Jueces",
    "📊 Métricas de Coordinación",
    "📈 Madurez + Costos",
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

            docs_citados = [k for k in d["kb"] if k in d["respuesta"]]
            if docs_citados:
                st.markdown("**Documentos citados:**")
                cols = st.columns(len(docs_citados))
                for i, did in enumerate(docs_citados):
                    txt = d["kb"][did]
                    c = "🔴 DESACTUALIZADO" if "DESACTUALIZADO" in txt else ("🟢 VIGENTE" if "VIGENTE" in txt else "⚪")
                    cols[i].markdown(f"**{c}**")
                    cols[i].caption(f"**{did}**: {txt[:60]}...")

            e = d["eval"]
            avg = sum([e["grounded"]["score"], e["behavioral"]["score"],
                       e["safety"]["score"], e["debate"]["score"]]) / 4
            verdict_color = "#22c55e" if avg >= umbral else ("#eab308" if avg >= 0.4 else "#ef4444")
            st.markdown(f"""<div class="card" style="border-color:{verdict_color}">
<b>Score promedio del consejo de jueces:</b>
<span class="score" style="color:{verdict_color}"> {avg:.0%}</span>
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
            sc = "#22c55e" if score >= umbral else ("#eab308" if score >= 0.4 else "#ef4444")
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
            for f in e["behavioral"]["flags"]:
                icon = "✅" if f.upper().startswith("OK") else "⚠️"
                st.markdown(f"{icon} {f}")
        with st.expander("🔴 Safety Judge — Riesgos"):
            st.markdown(f"**Acción:** {e['safety']['action']}")
            for issue in e["safety"]["issues"]:
                icon = "✅" if issue.upper().startswith("OK") else "🚨"
                st.markdown(f"{icon} {issue}")
        with st.expander("🟡 Debate Judge — Contraejemplos"):
            st.markdown(f"**Veredicto:** {e['debate']['verdict']}")
            for f in (e["debate"]["finds"] or ["Sin contraejemplos"]):
                st.markdown(f"🔍 {f}")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — MÉTRICAS DE COORDINACIÓN
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Métricas de Coordinación Multi-Agente")
    st.caption("Estas métricas son gratis una vez que tienes trazas OTEL. Miden el sistema, no al agente individual.")

    RUNS = [
        {"run":"Run-A (Normal)",           "agreement":0.89,"correction_yield":0.82,"handoff_fidelity":0.91,"tool_efficiency":0.75,"latency_coupling":0.22,"cascade_depth":0.0,"semantic_drift":0.08},
        {"run":"Run-B (Delegation Gap)",   "agreement":0.71,"correction_yield":0.30,"handoff_fidelity":0.55,"tool_efficiency":0.60,"latency_coupling":0.28,"cascade_depth":0.67,"semantic_drift":0.32},
        {"run":"Run-C (Tool Failure)",     "agreement":0.45,"correction_yield":0.10,"handoff_fidelity":0.40,"tool_efficiency":0.10,"latency_coupling":0.80,"cascade_depth":1.0, "semantic_drift":0.75},
        {"run":"Run-D (Stale Document)",   "agreement":0.68,"correction_yield":0.55,"handoff_fidelity":0.72,"tool_efficiency":0.70,"latency_coupling":0.25,"cascade_depth":0.33,"semantic_drift":0.35},
        {"run":"Run-E (Verif. Theater)",   "agreement":0.93,"correction_yield":0.05,"handoff_fidelity":0.88,"tool_efficiency":0.78,"latency_coupling":0.20,"cascade_depth":0.67,"semantic_drift":0.20},
    ]
    df = pd.DataFrame(RUNS).set_index("run")
    # Invertir métricas donde más alto = peor
    df_norm = df.copy()
    df_norm["latency_coupling"] = 1 - df_norm["latency_coupling"]
    df_norm["cascade_depth"]    = 1 - df_norm["cascade_depth"]
    df_norm["semantic_drift"]   = 1 - df_norm["semantic_drift"]

    metric_labels = ["Acuerdo\nJueces","Correcc.\nYield","Handoff\nFidelidad","Tool\nEfic.","Latencia\n(inv)","Cascade\n(inv)","Drift\nSem.(inv)"]

    fig_heat = go.Figure(go.Heatmap(
        z=df_norm.values,
        x=metric_labels,
        y=df_norm.index.tolist(),
        colorscale="RdYlGn", zmin=0, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in df_norm.values],
        texttemplate="%{text}",
        hovertemplate="%{y} | %{x}: %{z:.2f}<extra></extra>"
    ))
    fig_heat.update_layout(
        title="Heatmap de Coordinación (verde=bueno, rojo=problema)",
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        font=dict(color="#e2e8f0"), height=320,
        margin=dict(l=160,r=20,t=60,b=80)
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("""
> **🔍 Observa Run-E (Verification Theater):** `agreement = 0.93` → todos los jueces de acuerdo. Parece el mejor.
> Pero `correction_yield = 0.05` → casi nada de lo detectado se corrigió. El verificador dijo OK sin verificar realmente.
>
> **Run-C (Tool Failure):** todo en rojo — pero al menos el sistema **sabe** que falló. Es recuperable. Run-E es silencioso.
""")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — MADUREZ + COSTOS
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Modelo de Madurez: ¿Qué construir primero?")
    st.caption("5 etapas. No hay que llegar al Stage 4 desde el día 1. El mejor ROI está en Stage 2+3.")

    STAGES = [
        {"stage":0,"nombre":"Logs Only",          "semanas":1,"roi":5000, "fallos":10,"estado":"PARCIAL"},
        {"stage":1,"nombre":"Deterministic Gates", "semanas":2,"roi":18000,"fallos":25,"estado":"PENDIENTE"},
        {"stage":2,"nombre":"Grounded Judging",    "semanas":3,"roi":45000,"fallos":55,"estado":"PENDIENTE"},
        {"stage":3,"nombre":"Behavioral Scoring",  "semanas":3,"roi":55000,"fallos":75,"estado":"PENDIENTE"},
        {"stage":4,"nombre":"Council + Replay",    "semanas":6,"roi":70000,"fallos":92,"estado":"FUTURO"},
    ]

    # Métricas principales
    c0,c1,c2,c3,c4 = st.columns(5)
    for col, s in zip([c0,c1,c2,c3,c4], STAGES):
        color = "#eab308" if s["estado"]=="PARCIAL" else "#22c55e" if s["estado"]=="PENDIENTE" else "#475569"
        delta_txt = "← Estás aquí" if s["estado"]=="PARCIAL" else ("← Mejor ROI" if s["stage"] in [2,3] else "")
        col.markdown(f"""<div class="card" style="border-color:{color};text-align:center">
<div style="color:{color};font-size:.75rem;font-weight:bold">Stage {s['stage']}</div>
<div style="color:#e2e8f0;font-size:.8rem;margin:4px 0">{s['nombre']}</div>
<div style="color:{color};font-size:1.4rem;font-weight:bold">{s['fallos']}%</div>
<div style="color:#94a3b8;font-size:.7rem">fallos evitados</div>
<div style="color:{color};font-size:.7rem;margin-top:4px">{delta_txt}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    colors_s = ["#eab308" if s["estado"]=="PARCIAL" else "#22c55e" if s["estado"]=="PENDIENTE" else "#475569" for s in STAGES]
    fig_roi = go.Figure(go.Bar(
        x=[s["nombre"] for s in STAGES],
        y=[s["roi"] for s in STAGES],
        marker_color=colors_s,
        text=[f'${s["roi"]:,}<br>{s["semanas"]} sem' for s in STAGES],
        textposition="outside"
    ))
    fig_roi.update_layout(
        title="ROI mensual estimado por Stage (5,000 solicitudes/mes)",
        paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
        font=dict(color="#e2e8f0"), height=340,
        yaxis=dict(color="#94a3b8"),
        xaxis=dict(color="#94a3b8", tickangle=-15),
        margin=dict(l=20,r=20,t=60,b=80)
    )
    st.plotly_chart(fig_roi, use_container_width=True)

    st.markdown("""
| Tipo de evaluación | Costo por request | Latencia | Fallos detectados |
|---|---|---|---|
| Checks determinísticos | ~$0.0001 | 5 ms | 25% |
| Grounded Judge (LLM) | ~$0.008 | 320 ms | 55% |
| Behavioral Judge (LLM) | ~$0.003 | 80 ms | 70% |
| Safety Judge (LLM) | ~$0.004 | 120 ms | 80% |
| Debate Council (4×LLM) | ~$0.025 | 850 ms | 92% |
""")
    st.info("**Regla de oro:** Checks determinísticos siempre inline (5ms, gratis). Jueces LLM solo cuando agrega valor real. Stack completo para 5,000 req/mes: < $200/mes.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — CHAOS ENGINEERING
# ═════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("🔥 Chaos Engineering: inyectar fallos para entender la robustez")
    st.caption("Los mejores sistemas no son los que nunca fallan — son los que detectan y comunican el fallo. (doc.md sección 9.1)")

    CHAOS_TYPES = {
        "📄 Documento obsoleto":    {"desc":"DOC con fecha vieja aparece primero en el contexto","impact":"Stale Document → Juez detecta CONTRADICTED","score_delta":-0.55},
        "⚡ Fuentes conflictivas":  {"desc":"Dos docs con datos opuestos en la misma KB","impact":"Reasoning Drift → Debate Judge dice REVISE","score_delta":-0.45},
        "🔧 Tool failure":          {"desc":"El agente retriever devuelve vacío por error","impact":"Delegation Gap → Behavioral dice INCOMPLETE","score_delta":-0.65},
        "✂️ Contexto truncado":     {"desc":"El contexto se corta a la mitad por límite de tokens","impact":"Error Cascade → todos los scores caen","score_delta":-0.70},
    }

    BASE_SCORES = {"grounded":0.90,"behavioral":0.90,"safety":0.95,"debate":0.90}

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown("**Selecciona el tipo de fallo a inyectar:**")
        chaos_sel = st.radio("", list(CHAOS_TYPES.keys()), label_visibility="collapsed")
        info = CHAOS_TYPES[chaos_sel]
        st.markdown(f"""<div class="card" style="border-color:#f97316">
<b style="color:#f97316">{chaos_sel}</b><br>
<b>Qué hace:</b> {info['desc']}<br>
<b>Modo de fallo:</b> {info['impact']}
</div>""", unsafe_allow_html=True)

        inject_btn = st.button("💉 Inyectar fallo y evaluar", type="primary", use_container_width=True)

    with col_b:
        if inject_btn:
            delta = info["score_delta"]
            import random; random.seed(42)
            scores_after = {k: max(0, v + delta + random.uniform(-0.05,0.05))
                            for k,v in BASE_SCORES.items()}
            avg_before = sum(BASE_SCORES.values()) / 4
            avg_after  = sum(scores_after.values()) / 4

            st.markdown("**Comparación: antes vs después del fallo**")
            df_chaos = pd.DataFrame({
                "Juez":    ["Grounded","Behavioral","Safety","Debate","PROMEDIO"],
                "Sin fallo": [BASE_SCORES[k] for k in ["grounded","behavioral","safety","debate"]] + [avg_before],
                "Con fallo": [scores_after[k] for k in ["grounded","behavioral","safety","debate"]] + [avg_after],
            })
            df_chaos["Δ"] = (df_chaos["Con fallo"] - df_chaos["Sin fallo"]).map(lambda x: f"{x:+.2f}")
            df_chaos["Sin fallo"] = df_chaos["Sin fallo"].map(lambda x: f"{x:.0%}")
            df_chaos["Con fallo"] = df_chaos["Con fallo"].map(lambda x: f"{x:.0%}")
            st.dataframe(df_chaos, use_container_width=True, hide_index=True)

            fig_ch = go.Figure()
            judges_labels = ["Grounded","Behavioral","Safety","Debate"]
            before_vals = [BASE_SCORES[k] for k in ["grounded","behavioral","safety","debate"]]
            after_vals  = [scores_after[k] for k in ["grounded","behavioral","safety","debate"]]
            for vals, name, color in [(before_vals,"Sin fallo","#22c55e"),(after_vals,"Con fallo","#ef4444")]:
                v2 = vals + [vals[0]]
                t2 = judges_labels + [judges_labels[0]]
                fig_ch.add_trace(go.Scatterpolar(r=v2, theta=t2, fill="toself",
                    name=name, line=dict(color=color, width=2)))
            fig_ch.update_layout(
                polar=dict(radialaxis=dict(range=[0,1], color="#94a3b8"), bgcolor="#1e293b",
                           angularaxis=dict(color="#94a3b8")),
                paper_bgcolor="#0f172a", font=dict(color="#e2e8f0"),
                legend=dict(font=dict(color="#e2e8f0")),
                height=300, margin=dict(l=40,r=40,t=40,b=20)
            )
            st.plotly_chart(fig_ch, use_container_width=True)

            drop = avg_before - avg_after
            c1, c2 = st.columns(2)
            c1.metric("Score sin fallo", f"{avg_before:.0%}")
            c2.metric("Score con fallo", f"{avg_after:.0%}", delta=f"{-drop:.0%}", delta_color="inverse")
        else:
            st.markdown('<div class="card" style="border-color:#334155;color:#64748b">Selecciona un tipo de fallo y presiona el botón para ver cómo afecta los scores de los jueces.</div>', unsafe_allow_html=True)
