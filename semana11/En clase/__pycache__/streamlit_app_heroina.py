from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


DATA_PATH = Path(__file__).resolve().parent / "heroina-colombia.csv"
SPECIAL_TERRITORIES = {"ECUADOR", "AGUAS INTERNACIONALES", "SIN ESTABLECER"}


st.set_page_config(
    page_title="Dashboard de Heroina en Colombia",
    page_icon=":bar_chart:",
    layout="wide",
)


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.rename(
        columns={
            "fecha_hecho": "fecha",
            "departamento": "departamento",
            "municipio": "municipio",
            "cantidad": "cantidad_kg",
            "unidad": "unidad",
        }
    )
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["departamento"] = df["departamento"].astype(str).str.strip()
    df["municipio"] = df["municipio"].astype(str).str.strip()
    df["cantidad_kg"] = pd.to_numeric(df["cantidad_kg"], errors="coerce")
    df["anio"] = df["fecha"].dt.year
    df["mes"] = df["fecha"].dt.to_period("M").dt.to_timestamp()
    df["territorio_especial"] = df["departamento"].isin(SPECIAL_TERRITORIES)
    return df.dropna(subset=["fecha", "cantidad_kg"]).sort_values("fecha")


df = load_data()
latest_date = df["fecha"].max().date()
valid_df = df.loc[~df["territorio_especial"]].copy()
total_valid_kg = valid_df["cantidad_kg"].sum()
top_department_global = (
    valid_df.groupby("departamento", as_index=False)["cantidad_kg"]
    .sum()
    .sort_values("cantidad_kg", ascending=False)
    .iloc[0]
)
top_department_share = top_department_global["cantidad_kg"] / total_valid_kg


st.title("Incautacion de heroina en Colombia")
st.write(
    "Dashboard interactivo basado en el dataset oficial "
    "`INCAUTACION DE HEROINA` de datos.gov.co. "
    "Pregunta guia: donde y cuando se concentra la mayor cantidad incautada?"
)
st.caption(
    f"Corte de datos: del {df['fecha'].min().date()} al {latest_date}. "
    "El ano 2026 aparece parcial porque el ultimo registro disponible llega hasta "
    f"{latest_date}."
)

st.info(
    "Insight inicial: al excluir territorios especiales, "
    f"{top_department_global['departamento']} concentra "
    f"{top_department_share:.1%} del total incautado registrado."
)


with st.sidebar:
    st.header("Filtros")

    include_special = st.checkbox(
        "Incluir territorios especiales",
        value=False,
        help="Agrega ECUADOR, AGUAS INTERNACIONALES y SIN ESTABLECER.",
    )

    base_df = df if include_special else valid_df
    available_departments = sorted(base_df["departamento"].unique())

    selected_departments = st.multiselect(
        "Departamentos",
        options=available_departments,
        default=available_departments,
    )

    year_range = st.slider(
        "Rango de anos",
        min_value=int(base_df["anio"].min()),
        max_value=int(base_df["anio"].max()),
        value=(int(base_df["anio"].min()), int(base_df["anio"].max())),
    )

    quantity_range = st.slider(
        "Cantidad por registro (kg)",
        min_value=0.0,
        max_value=float(base_df["cantidad_kg"].max()),
        value=(0.0, float(base_df["cantidad_kg"].max())),
    )

    top_n = st.slider("Top categorias en barras", min_value=5, max_value=15, value=10)
    show_table = st.checkbox("Mostrar tabla filtrada", value=False)


if not selected_departments:
    st.warning("Selecciona al menos un departamento para continuar.")
    st.stop()


filtered_df = base_df[
    base_df["departamento"].isin(selected_departments)
    & base_df["anio"].between(year_range[0], year_range[1])
    & base_df["cantidad_kg"].between(quantity_range[0], quantity_range[1])
].copy()

if filtered_df.empty:
    st.warning("Los filtros actuales no devuelven registros. Ajustalos para ver resultados.")
    st.stop()


metric_col1, metric_col2, metric_col3 = st.columns(3)

total_kg = filtered_df["cantidad_kg"].sum()
total_records = len(filtered_df)
avg_per_record = filtered_df["cantidad_kg"].mean()

with metric_col1:
    st.metric("Total incautado", f"{total_kg:,.2f} kg")

with metric_col2:
    st.metric("Registros", f"{total_records:,}")

with metric_col3:
    st.metric("Promedio por registro", f"{avg_per_record:,.3f} kg")


top_filtered_department = (
    filtered_df.groupby("departamento", as_index=False)["cantidad_kg"]
    .sum()
    .sort_values("cantidad_kg", ascending=False)
    .iloc[0]
)
share_filtered = top_filtered_department["cantidad_kg"] / total_kg

st.markdown(
    f"""
**Lectura rapida:** entre **{year_range[0]}** y **{year_range[1]}**, el departamento
con mayor volumen dentro del filtro es **{top_filtered_department['departamento']}**,
con **{top_filtered_department['cantidad_kg']:,.2f} kg**, equivalente al
**{share_filtered:.1%}** del total visible.
"""
)


chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Tendencia anual")

    yearly_df = (
        filtered_df.groupby("anio", as_index=False)
        .agg(total_kg=("cantidad_kg", "sum"), registros=("cantidad_kg", "size"))
        .sort_values("anio")
    )

    fig_trend = px.line(
        yearly_df,
        x="anio",
        y="total_kg",
        markers=True,
        title="Total incautado por ano",
        hover_data={"registros": True, "total_kg": ":.2f"},
    )
    fig_trend.update_layout(xaxis_title="", yaxis_title="Kilogramos")
    st.plotly_chart(fig_trend, use_container_width=True)

with chart_col2:
    st.subheader("Top departamentos")

    department_df = (
        filtered_df.groupby("departamento", as_index=False)["cantidad_kg"]
        .sum()
        .sort_values("cantidad_kg", ascending=False)
        .head(top_n)
    )

    fig_bar = px.bar(
        department_df,
        x="departamento",
        y="cantidad_kg",
        color="departamento",
        title="Departamentos con mayor cantidad incautada",
        text_auto=".2f",
    )
    fig_bar.update_layout(showlegend=False, xaxis_title="", yaxis_title="Kilogramos")
    st.plotly_chart(fig_bar, use_container_width=True)


st.subheader("Distribucion por departamento")

distribution_df = (
    filtered_df.groupby(["departamento", "anio"], as_index=False)["cantidad_kg"]
    .sum()
    .sort_values(["departamento", "anio"])
)

fig_box = px.box(
    distribution_df,
    x="departamento",
    y="cantidad_kg",
    points="all",
    color="departamento",
    title="Variabilidad anual del total incautado por departamento",
)
fig_box.update_layout(showlegend=False, xaxis_title="", yaxis_title="Kilogramos")
st.plotly_chart(fig_box, use_container_width=True)


download_df = filtered_df.sort_values(["fecha", "departamento", "municipio"]).copy()
download_df["fecha"] = download_df["fecha"].dt.strftime("%Y-%m-%d")

st.download_button(
    label="Descargar datos filtrados en CSV",
    data=download_df.to_csv(index=False).encode("utf-8"),
    file_name="heroina_colombia_filtrado.csv",
    mime="text/csv",
)


if show_table:
    st.subheader("Datos filtrados")
    st.dataframe(
        download_df[
            [
                "fecha",
                "departamento",
                "municipio",
                "cantidad_kg",
                "unidad",
                "territorio_especial",
            ]
        ],
        use_container_width=True,
    )
