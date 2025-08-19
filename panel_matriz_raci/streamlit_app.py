# streamlit_app.py — versión con hero eliminado y contadores abajo
# Ejecuta: streamlit run streamlit_app.py

from __future__ import annotations
import re
import base64
import unicodedata
from io import BytesIO
from datetime import datetime
from mimetypes import guess_type
from pathlib import Path

import pandas as pd
import streamlit as st

# --- Optional: use AG Grid if available (mobile-friendly table) ---
try:
    from st_aggrid import AgGrid, GridOptionsBuilder, JsCode, GridUpdateMode
    AGGRID_AVAILABLE = True
except Exception:
    AGGRID_AVAILABLE = False

# -------------------- Configuración general --------------------
def _find_logo_path():
    candidates = [
        Path("logo_2024.ico"), Path("logo_2024.png"), Path("logo_2024.jpg"), Path("logo_2024"),
        Path("Metro Santiago.jpg"), Path("Metro Santiago.JPG"), Path("logo_metro_versiones-05.jpg"),
        Path("assets/metro_logo.png"), Path("assets/metro_logo.jpg")
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None

_page_icon = _find_logo_path()
st.set_page_config(page_title="Matriz RACI", page_icon=_page_icon or "🅼", layout="wide")

COLOR_PRIMARIO = "#E10600"  # rojo Metro
COLOR_BORDE = "#e5e7eb"
COLOR_TXT_SUAVE = "#6b7280"

# -------------------- Estilos --------------------
st.markdown(
    f"""
<style>
  .block-container {{ padding-top: 92px; padding-bottom: 1rem; }}
  h1, h2, h3, h4 {{ font-weight: 800; letter-spacing: .2px; }}

  /* TOP BAR */
  .metro-topbar {{ position: sticky; top: 0; z-index: 1000; background: linear-gradient(90deg, #E10600 0%, #C20A0A 50%, #9E0B0B 100%); color: #fff; border-bottom: 1px solid #9e0b0b; box-shadow: 0 4px 10px rgba(0,0,0,.18); }}
  .metro-topbar .wrap {{ max-width: 1200px; margin: 0 auto; display: flex; align-items: center; justify-content: space-between; padding: 8px 20px; }}
  .metro-topbar .brand {{ display:flex; align-items:center; gap:.6rem; }}
  .metro-topbar .brand img {{ height: 28px; }}
  .metro-topbar .name {{ font-weight:800; letter-spacing:.3px; }}
  .matriz-title {{ display:inline-block; font-weight:800; letter-spacing:.3px; border:1px solid rgba(255,255,255,.45); background: rgba(255,255,255,.16); padding: 8px 16px; border-radius: 999px; box-shadow: inset 0 0 0 1px rgba(255,255,255,.08); }}

  /* Chips y métricas */
  .metro-chip {{ display:inline-flex; align-items:center; gap:.5rem; font-weight:700; border:1px solid {COLOR_BORDE}; border-radius:999px; padding:.35rem .8rem; background:#fff; box-shadow: 0 1px 2px rgba(0,0,0,.04); }}
  .dot {{ width:.75rem; height:.75rem; border-radius:50%; display:inline-block; }}
  .A {{ background:#ef4444; }}
  .R {{ background:#10b981; }}
  .C {{ background:#f59e0b; }}
  .I {{ background:#3b82f6; }}
  .AR {{ background:linear-gradient(90deg,#ef4444,#10b981); }}

  .metric {{ border:1px solid {COLOR_BORDE}; border-radius:16px; padding:12px 14px; background:#fff; text-align:center; }}
  .metric .num {{ font-size:1.6rem; font-weight:900; }}
  .metric .lbl {{ color:{COLOR_TXT_SUAVE}; font-weight:600; letter-spacing:.5px; }}

  .stDataFrame {{ border:1px solid {COLOR_BORDE}; border-radius: 12px; }}

  /* AG Grid theme tweaks */
  .ag-theme-alpine {{
    --ag-header-background-color: #fafafa;
    --ag-odd-row-background-color: #ffffff;
    --ag-row-border-color: {COLOR_BORDE};
    --ag-header-foreground-color: #111827;
    --ag-selected-row-background-color: #fff7ed;
    --ag-font-size: 14px;
    border: 1px solid {COLOR_BORDE};
    border-radius: 12px;
  }}
  .ag-theme-alpine .ag-cell-wrap-text {{ white-space: normal !important; }}

  @media (max-width: 640px) {{
    .block-container {{ padding-left: 10px; padding-right: 10px; padding-top: 80px; }}
    .metro-topbar .wrap {{ padding: 6px 12px; }}
  }}
</style>
""",
    unsafe_allow_html=True,
)

# -------------------- Top bar con logo --------------------

def _logo_data_url() -> str:
    candidates = [
        Path("logo_2024.ico"), Path("logo_2024.png"), Path("logo_2024.jpg"), Path("logo_2024"),
        Path("Metro Santiago.jpg"), Path("Metro Santiago.JPG"), Path("logo_metro_versiones-05.jpg"),
        Path("assets/metro_logo.png"), Path("assets/metro_logo.jpg")
    ]
    for p in candidates:
        if p.exists():
            mime, _ = guess_type(p.name)
            mime = mime or ("image/png" if p.suffix.lower()==".png" else "image/jpeg")
            data = base64.b64encode(p.read_bytes()).decode()
            return f"data:{mime};base64,{data}"
    return ""

_logo = _logo_data_url()
_logo_tag = f"<img src='{_logo}' alt='Metro'/>" if _logo else ""

st.markdown(
    f"""
<div class='metro-topbar'>
  <div class='wrap'>
    <div class='brand'>
      {_logo_tag}
      <span class='name'>Metro de Santiago</span>
    </div>
    <span class='matriz-title'>Matriz RACI</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# -------------------- Datos --------------------
@st.cache_data
def load_data(path: Path, sheet: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet)
    df.columns = df.columns.str.strip()
    return df

XLSX_PATH = Path("Matriz RACI Nico.xlsx")
SHEET = "streamlit"  # Ajusta si tu hoja tiene otro nombre

df = load_data(XLSX_PATH, SHEET)

# -------------------- Descubrimiento de columnas RACI --------------------
BASE_COLS = [c for c in ["Proceso", "Tarea"] if c in df.columns]
RACI_PAT = re.compile(r"^[ARCI/\-\s]*$")
_cands = [c for c in df.columns if c not in BASE_COLS]
area_cols = [c for c in _cands if ({str(v).strip().upper() for v in df[c].dropna().unique()} and all(RACI_PAT.match(str(v).strip().upper() or "") for v in df[c].dropna().unique()))] or _cands

# -------------------- Sidebar filtros --------------------
with st.sidebar:
    st.header("Filtros")
    area = st.selectbox("Área (columna)", area_cols, index=0)
    roles_all = ["A", "R", "C", "I"]
    roles_sel = st.multiselect("Rol RACI", roles_all, default=roles_all)
    q = st.text_input("Buscar en Proceso/Tarea", placeholder="Palabra clave…")

    st.markdown(
        """
        **Leyenda**<br>
        <span class='metro-chip'><span class='dot A'></span>A (Accountable)</span>
        <span class='metro-chip'><span class='dot R'></span>R (Responsible)</span>
        <span class='metro-chip'><span class='dot C'></span>C (Consulted)</span>
        <span class='metro-chip'><span class='dot I'></span>I (Informed)</span>
        """,
        unsafe_allow_html=True,
    )

    # Filtro Proceso (desde df completo para que siempre estén todas las opciones)
    proc_sel = None
    if 'Proceso' in df.columns:
        proc_series_full = (
            df['Proceso'].dropna().astype(str).str.strip()
        )
        proc_options = ['Todos los procesos'] + list(pd.unique(proc_series_full))
        default_proc = st.session_state.get('proc_sel', proc_options[1] if len(proc_options) > 1 else proc_options[0])
        default_idx = proc_options.index(default_proc) if default_proc in proc_options else 0
        proc_sel = st.selectbox('Proceso', proc_options, index=default_idx, key='proc_sel', help='Filtra la tabla y las métricas por proceso')

# -------------------- Lógica de filtrado --------------------
ROLE_RE = re.compile(r"[ARCI]")

def parse_roles(cell: str) -> set[str]:
    if cell is None:
        return set()
    return set(ROLE_RE.findall(str(cell).upper()))

work = df.copy()
work["Rol"] = (
    work[area].fillna("").astype(str).str.upper().str.replace(" ", "", regex=False)
)

# Filtrar por roles
if roles_sel:
    sel = set(roles_sel)
    mask_roles = work["Rol"].apply(lambda x: bool(parse_roles(x)) and parse_roles(x).issubset(sel))
    work = work[mask_roles]
else:
    work = work.iloc[0:0]

# Filtrar por proceso (si aplica)
if proc_sel and proc_sel != 'Todos los procesos' and 'Proceso' in work.columns:
    work = work[work['Proceso'].astype(str).str.strip() == proc_sel]

# Filtro de búsqueda
if q:
    cols = [c for c in ["Proceso", "Tarea"] if c in work.columns]
    if cols:
        m = False
        for c in cols:
            m = m | work[c].astype(str).str.contains(q, case=False, na=False)
        work = work[m]

# -------------------- Cálculo de métricas --------------------
count_A  = int(work["Rol"].apply(lambda x: "A" in parse_roles(x)).sum())
count_R  = int(work["Rol"].apply(lambda x: "R" in parse_roles(x)).sum())
count_C  = int(work["Rol"].apply(lambda x: "C" in parse_roles(x)).sum())
count_I  = int(work["Rol"].apply(lambda x: "I" in parse_roles(x)).sum())
count_AR = int(work["Rol"].apply(lambda x: parse_roles(x) == {"A","R"}).sum())

# -------------------- Tabla --------------------
st.subheader(f"Tareas para **{area}**")
if proc_sel:
    st.caption(f"Proceso seleccionado: {proc_sel}")

view = work.loc[:, [c for c in ["Tarea", area] if c in work.columns]].rename(columns={area: "Rol (RACI)"})

# Ordenar alfabéticamente por Tarea (ignorando tildes y mayúsculas)
def _sort_key_series(s: pd.Series) -> pd.Series:
    return s.astype(str).map(lambda x: ''.join(c for c in unicodedata.normalize('NFD', x) if unicodedata.category(c) != 'Mn').lower())

if 'Tarea' in view.columns:
    view = view.sort_values(by='Tarea', key=_sort_key_series, kind='mergesort').reset_index(drop=True)
# --- Tabla con AG Grid (responsive y mobile-friendly) ---
if AGGRID_AVAILABLE:
    gb = GridOptionsBuilder.from_dataframe(view)
    gb.configure_default_column(resizable=True, sortable=True, filter=True, wrapText=True, autoHeight=True)
    gb.configure_grid_options(domLayout='normal', quickFilterText=(q or ""), rowHeight=36, headerHeight=38, suppressPaginationPanel=True)
    gb.configure_column('Tarea', header_name='Tarea', flex=3, minWidth=280, cellStyle={'white-space':'normal'}, sort='asc')
    raci_renderer = JsCode("""
class RaciRenderer {
  init(params){
    const val = (params.value || '').toString().toUpperCase().split(' ').join('');
    const map = { 'A':'#ef4444','R':'#10b981','C':'#f59e0b','I':'#3b82f6' };
    const dot = function(c){
      return '<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:'+c+';margin-right:6px;"></span>';
    };
    let html;
    if (val==='A/R' || val==='AR' || val==='A,R'){
      html = dot('#ef4444') + dot('#10b981') + 'A/R';
    } else {
      const color = map[val] || '#9ca3af';
      html = dot(color) + val;
    }
    this.eGui = document.createElement('span');
    this.eGui.innerHTML = html;
  }
  getGui(){ return this.eGui; }
}
""")
    gb.configure_column('Rol (RACI)', header_name='Rol', width=100, minWidth=84, maxWidth=140, pinned='right', cellRenderer=raci_renderer, wrapText=True, autoHeight=True)
    go = gb.build()
    AgGrid(
        view,
        height=420,
        gridOptions=go,
        theme='alpine',
        fit_columns_on_grid_load=True,
        allow_unsafe_jscode=True,
        enable_enterprise_modules=False,
        update_mode=GridUpdateMode.NO_UPDATE,
    )
else:
    st.dataframe(view, use_container_width=True, hide_index=True)

# -------------------- CONTADORES (movidos ABAJO) --------------------
col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1])
col1.markdown(f"""
<div class='metric'>
  <div class='lbl'><span class='dot A'></span> A</div>
  <div class='num'>{count_A}</div>
</div>""", unsafe_allow_html=True)
col2.markdown(f"""
<div class='metric'>
  <div class='lbl'><span class='dot R'></span> R</div>
  <div class='num'>{count_R}</div>
</div>""", unsafe_allow_html=True)
col3.markdown(f"""
<div class='metric'>
  <div class='lbl'><span class='dot C'></span> C</div>
  <div class='num'>{count_C}</div>
</div>""", unsafe_allow_html=True)
col4.markdown(f"""
<div class='metric'>
  <div class='lbl'><span class='dot I'></span> I</div>
  <div class='num'>{count_I}</div>
</div>""", unsafe_allow_html=True)
if {"A","R"}.issubset(set(roles_sel)):
    col5.markdown(f"""
    <div class='metric'>
      <div class='lbl'><span class='dot AR'></span> A/R</div>
      <div class='num'>{count_AR}</div>
    </div>""", unsafe_allow_html=True)
else:
    col5.empty()

# -------------------- Descarga XLSX (al final) --------------------

def _slugify(text: str) -> str:
    norm = ''.join(c for c in unicodedata.normalize('NFD', str(text)) if unicodedata.category(c) != 'Mn')
    norm = norm.lower().strip()
    norm = re.sub(r"[^a-z0-9]+", "_", norm)
    norm = re.sub(r"_+", "_", norm).strip('_')
    return norm or 'sin_nombre'

proc_for_name = proc_sel if (proc_sel and proc_sel != 'Todos los procesos') else 'Todos los procesos'
filename = f"{datetime.now():%Y%m%d}_{_slugify(proc_for_name)}_{_slugify(area)}.xlsx"

bio = BytesIO()
with pd.ExcelWriter(bio, engine='openpyxl') as writer:
    view.to_excel(writer, index=False, sheet_name='raci')
bio.seek(0)

# Espacio para el botón de descarga
st.markdown("---")  # Línea separadora

st.download_button(
    label="Descargar XLSX",
    data=bio,
    file_name=filename,
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
