from __future__ import annotations

import streamlit as st

from utils.helpers import apply_custom_css, render_top_navbar
from utils.sqlite_db import build_sqlite_db, db_exists, list_analysis_queries, run_query

st.set_page_config(page_title="SQL Analytics", layout="wide")
apply_custom_css()
render_top_navbar()

st.title("SQL Analytics")
st.caption("Edit and run SQL against SQLite. Templates are starting points — change anything before you run.")

if not db_exists():
    st.warning("SQLite database not built yet.")
    if st.button("Build database from CSV files", type="primary"):
        with st.spinner("Loading CSVs into SQLite (first run can take a few minutes)..."):
            build_sqlite_db(force=True)
        st.success("Database ready.")
        st.rerun()
    st.code("python -m pipelines.build_sqlite_db", language="bash")
    st.stop()

queries = list_analysis_queries()
template_titles = [q["title"] for q in queries]
template_map = {q["title"]: q["sql"] for q in queries}

if "sql_editor" not in st.session_state:
    st.session_state.sql_editor = template_map.get(template_titles[0], "SELECT 1;") if template_titles else "SELECT 1;"
if "template_select" not in st.session_state:
    st.session_state.template_select = template_titles[0] if template_titles else "Custom query"


def load_template() -> None:
    title = st.session_state.template_select
    if title == "Custom query":
        return
    st.session_state.sql_editor = template_map[title]


with st.sidebar:
    st.header("Schema")
    st.markdown(
        """
**Tables:** `results`, `goalscorers`, `shootouts`, `players`, `appearances`, `games`, `clubs`, `competitions`

**Views:** `player_season_stats`, `international_match_outcomes`
        """
    )

toolbar = st.columns([2, 1, 1])
with toolbar[0]:
    st.selectbox(
        "Load template",
        options=(["Custom query"] + template_titles) if template_titles else ["Custom query"],
        key="template_select",
        on_change=load_template,
    )
with toolbar[1]:
    if st.button("Reset to template"):
        load_template()
        st.rerun()
with toolbar[2]:
    run_clicked = st.button("Run query", type="primary", use_container_width=True)

st.text_area(
    "SQL editor",
    key="sql_editor",
    height=320,
    placeholder="SELECT * FROM results LIMIT 10;",
    label_visibility="collapsed",
)

if run_clicked:
    sql = st.session_state.sql_editor.strip()
    if not sql:
        st.warning("Enter a SQL query first.")
    else:
        with st.spinner("Running..."):
            try:
                df = run_query(sql)
            except Exception as exc:
                st.error(str(exc))
            else:
                st.markdown(f"**{len(df):,} rows**")
                st.dataframe(df, use_container_width=True)

with st.expander("Database info"):
    tables = run_query(
        "SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view') ORDER BY type, name"
    )
    st.dataframe(tables, use_container_width=True)

    picked = st.selectbox("Preview table/view", tables["name"].tolist(), key="schema_preview")
    if picked:
        preview = run_query(f"SELECT * FROM {picked} LIMIT 5")
        st.dataframe(preview, use_container_width=True)
