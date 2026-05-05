from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "notebooks" / "01_analisis_conflictos_ucdp.ipynb"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip().splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip().splitlines(True),
    }


cells = [
    markdown(
        """
        # Intensidad y dano civil en conflictos recientes

        **Pregunta guia:** ¿como ha cambiado la intensidad de los conflictos armados recientes desde 2022, que paises concentran la mayor letalidad y que tanto predice la frecuencia mensual de eventos el numero de muertes estimadas?

        Este notebook esta pensado como entrega academica: cada bloque explica que se hace, por que se hace y que decision analitica queda documentada. La base cruda se toma de UCDP, se limpia en Python/Jupyter y se transforma en datos listos para graficas, storytelling y una app de Streamlit.
        """
    ),
    markdown(
        """
        ## 1. Contexto y objetivo

        UCDP registra eventos de violencia organizada con fecha, lugar, actores y estimaciones de muertes. Para el trabajo nos interesa demostrar el ciclo completo visto en clase: inspeccion inicial, limpieza, filtros, `groupby`, EDA univariado, comparaciones bivariadas, visualizacion, storytelling y regresion lineal.

        **Objetivo analitico:** transformar una base grande y parcialmente sucia en una explicacion clara sobre intensidad, dano civil e incertidumbre en conflictos actuales.
        """
    ),
    code(
        """
        from pathlib import Path
        import sys
        import site

        PROJECT_ROOT = Path.cwd()
        if PROJECT_ROOT.name == "notebooks":
            PROJECT_ROOT = PROJECT_ROOT.parent

        sys.path.append(str(PROJECT_ROOT / "src"))
        site.addsitedir(str(PROJECT_ROOT.parent / ".pydeps"))

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import plotly.express as px

        from ucdp_pipeline import (
            ensure_data_files,
            load_raw_events,
            initial_quality_report,
            clean_events,
            build_monthly_panel,
            build_global_monthly,
            linear_regression_summary,
            export_processed,
        )

        DATA_DIR = PROJECT_ROOT / "data"
        pd.set_option("display.max_columns", 80)
        sns.set_theme(style="whitegrid", palette="Set2")
        """
    ),
    markdown(
        """
        ## 2. Carga de datos

        El proyecto integra tres archivos oficiales:

        - GED 25.1: eventos globales entre 1989 y 2024.
        - Candidate GED 2025: eventos candidatos acumulados para 2025.
        - Candidate GED 26.0.3: eventos candidatos disponibles hasta marzo de 2026.

        Si los archivos no estan descargados, la siguiente celda puede descargarlos desde UCDP. En clase es recomendable ejecutar con internet antes de la presentacion y luego trabajar con los archivos locales.
        """
    ),
    code(
        """
        missing = ensure_data_files(DATA_DIR, download=False)
        if missing:
            print("Faltan archivos raw. Para descargarlos automaticamente ejecute:")
            print("ensure_data_files(DATA_DIR, download=True)")
            for path in missing:
                print("-", path)
        else:
            print("Archivos raw disponibles.")
        """
    ),
    code(
        """
        raw = load_raw_events(DATA_DIR)
        print(f"Filas crudas: {raw.shape[0]:,}")
        print(f"Columnas crudas: {raw.shape[1]:,}")
        display(raw.head())
        """
    ),
    markdown(
        """
        ## 3. Diagnostico inicial de calidad

        Antes de limpiar, se aplica el ritual de inspeccion de clase: dimensiones, tipos, nulos, duplicados y valores especiales. Este paso evita limpiar a ciegas.
        """
    ),
    code(
        """
        print("Rango de anos:", int(raw["year"].min()), "-", int(raw["year"].max()))
        print("Duplicados por id:", raw.duplicated("id").sum())
        display(raw.dtypes.to_frame("tipo").head(20))

        quality_raw = initial_quality_report(raw)
        display(quality_raw.head(15))
        """
    ),
    code(
        """
        sentinel_sources = (pd.to_numeric(raw["number_of_sources"], errors="coerce") == -1).sum()
        print("Registros con number_of_sources = -1:", int(sentinel_sources))

        fatality_cols = ["best", "high", "low", "deaths_civilians"]
        display(raw[fatality_cols].describe().T)
        """
    ),
    markdown(
        """
        ## 4. Limpieza y transformacion

        Decisiones aplicadas:

        - Convertir fechas a tipo datetime.
        - Convertir variables numericas con `pd.to_numeric`.
        - Eliminar duplicados por `id`.
        - Tratar `number_of_sources = -1` como nulo porque no es un conteo comparable.
        - Crear etiquetas legibles para tipos de violencia y precision.
        - Crear variables nuevas: mes, duracion, incertidumbre y proporcion de muertes civiles.
        - Validar coordenadas basicas.
        """
    ),
    code(
        """
        clean = clean_events(raw)
        print(f"Filas limpias: {clean.shape[0]:,}")
        print("Duplicados removidos:", clean.attrs.get("duplicates_removed", 0))
        display(clean.head())
        """
    ),
    code(
        """
        quality_clean = initial_quality_report(clean)
        display(quality_clean.head(15))

        comparison = pd.DataFrame({
            "metrica": ["filas", "duplicados_id", "columnas"],
            "antes": [len(raw), raw.duplicated("id").sum(), raw.shape[1]],
            "despues": [len(clean), clean.duplicated("id").sum(), clean.shape[1]],
        })
        display(comparison)
        """
    ),
    markdown(
        """
        ## 5. Foco reciente y preguntas de analisis

        Para contar una historia relacionada con guerras actuales, el analisis descriptivo se concentra en 2022-2026. Esto incluye Rusia-Ucrania, Israel-Palestina, Sudan, Etiopia y otros conflictos con alta letalidad reciente.
        """
    ),
    code(
        """
        recent = clean[clean["year"] >= 2022].copy()
        print(f"Eventos recientes desde 2022: {len(recent):,}")
        print(f"Muertes estimadas recientes: {recent['best'].sum():,.0f}")
        print(f"Muertes civiles recientes: {recent['deaths_civilians'].sum():,.0f}")
        """
    ),
    code(
        """
        by_year = recent.groupby("year").agg(
            eventos=("id", "count"),
            muertes=("best", "sum"),
            civiles=("deaths_civilians", "sum"),
        )
        by_year["proporcion_civil"] = by_year["civiles"] / by_year["muertes"]
        display(by_year)
        """
    ),
    markdown(
        """
        ## 6. Analisis univariado

        La variable `best` mide la mejor estimacion de muertes por evento. Su distribucion suele ser asimetrica: muchos eventos tienen pocas muertes y pocos eventos concentran una letalidad extrema. Por eso conviene mirar media, mediana, IQR y outliers.
        """
    ),
    code(
        """
        x = recent["best"].dropna()
        q1, q3 = x.quantile([0.25, 0.75])
        iqr = q3 - q1
        upper = q3 + 1.5 * iqr
        outliers = (x > upper).sum()

        summary_best = pd.DataFrame({
            "media": [x.mean()],
            "mediana": [x.median()],
            "desviacion": [x.std()],
            "iqr": [iqr],
            "limite_outlier_superior": [upper],
            "outliers": [outliers],
            "porcentaje_outliers": [outliers / len(x) * 100],
        })
        display(summary_best.round(2))
        """
    ),
    code(
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.histplot(np.log1p(recent["best"]), bins=40, ax=axes[0], color="#246A73")
        axes[0].set_title("Distribucion log(1 + muertes estimadas)")
        axes[0].set_xlabel("log(1 + best)")

        sns.boxplot(x=np.log1p(recent["best"]), ax=axes[1], color="#D9A441")
        axes[1].set_title("Boxplot logaritmico de letalidad por evento")
        axes[1].set_xlabel("log(1 + best)")
        plt.tight_layout()
        plt.show()
        """
    ),
    markdown(
        """
        ## 7. Agrupaciones: paises, conflictos y tipos de violencia

        Aqui reutilizamos `groupby` para pasar de registros individuales a hallazgos comparables. Esta parte construye los rankings que luego alimentan el storytelling.
        """
    ),
    code(
        """
        top_countries = (
            recent.groupby("country", dropna=False)
            .agg(eventos=("id", "count"), muertes=("best", "sum"), civiles=("deaths_civilians", "sum"))
            .sort_values("muertes", ascending=False)
            .head(15)
        )
        top_countries["proporcion_civil"] = top_countries["civiles"] / top_countries["muertes"]
        display(top_countries)
        """
    ),
    code(
        """
        top_conflicts = (
            recent.groupby("conflict_name", dropna=False)
            .agg(eventos=("id", "count"), muertes=("best", "sum"), civiles=("deaths_civilians", "sum"))
            .sort_values("muertes", ascending=False)
            .head(15)
        )
        top_conflicts["muertes_por_evento"] = top_conflicts["muertes"] / top_conflicts["eventos"]
        display(top_conflicts)
        """
    ),
    code(
        """
        fig = px.bar(
            top_countries.reset_index(),
            x="muertes",
            y="country",
            color="proporcion_civil",
            orientation="h",
            title="Paises con mayor letalidad estimada desde 2022",
            labels={"muertes": "Muertes estimadas", "country": "Pais", "proporcion_civil": "Proporcion civil"},
            color_continuous_scale="Tealrose",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        fig.show()
        """
    ),
    markdown(
        """
        ## 8. Panel mensual para visualizacion y regresion

        Se agregan eventos por mes, pais, conflicto y tipo de violencia. Este panel permite crear graficas temporales y ajustar una regresion lineal mensual.
        """
    ),
    code(
        """
        monthly_panel = build_monthly_panel(clean, min_year=2022)
        global_monthly = build_global_monthly(monthly_panel)
        display(monthly_panel.head())
        display(global_monthly.head())
        """
    ),
    code(
        """
        fig = px.line(
            global_monthly,
            x="event_month",
            y=["fatalities_best", "civilian_fatalities"],
            title="Letalidad mensual global en conflictos recientes",
            labels={"value": "Muertes", "event_month": "Mes", "variable": "Serie"},
        )
        fig.show()
        """
    ),
    markdown(
        """
        ## 9. Regresion lineal

        La regresion evalua una pregunta sencilla: cuando sube el numero mensual de eventos, ¿suben tambien las muertes estimadas? El modelo no pretende explicar toda la complejidad politica o militar; sirve para cuantificar una relacion inicial y discutir sus limites.
        """
    ),
    code(
        """
        regression_metrics = linear_regression_summary(global_monthly, target="fatalities_best")
        regression_metrics
        """
    ),
    code(
        """
        x = global_monthly["events"].to_numpy(dtype=float)
        y_hat = regression_metrics["intercept"] + regression_metrics["slope"] * x

        fig = px.scatter(
            global_monthly,
            x="events",
            y="fatalities_best",
            hover_data=["event_month"],
            title="Regresion lineal: eventos mensuales vs muertes estimadas",
            labels={"events": "Eventos por mes", "fatalities_best": "Muertes estimadas por mes"},
        )
        fig.add_scatter(x=global_monthly["events"], y=y_hat, mode="lines", name="Regresion lineal")
        fig.show()

        print(f"Pendiente: {regression_metrics['slope']:.2f}")
        print(f"R2: {regression_metrics['r2']:.3f}")
        print(f"RMSE: {regression_metrics['rmse']:.0f}")
        """
    ),
    markdown(
        """
        ## 10. Interpretacion para storytelling

        Lectura sugerida:

        1. La base reciente supera cien mil eventos, lo que justifica un flujo de limpieza y agregacion.
        2. La letalidad se concentra en pocos paises y conflictos, especialmente Rusia-Ucrania dentro del periodo observado.
        3. La regresion permite ver si la frecuencia de eventos explica la letalidad mensual, pero los residuales recuerdan que el contexto del conflicto importa.
        """
    ),
    code(
        """
        outputs = export_processed(DATA_DIR, clean, min_year=2022)
        for name, path in outputs.items():
            print(name, "->", path)
        """
    ),
    markdown(
        """
        ## 11. Conclusiones y limites

        **Conclusion principal:** contar eventos no basta para entender una guerra. La frecuencia mensual ayuda, pero la intensidad por evento, el dano civil, la incertidumbre de las estimaciones y la precision geografica cambian la historia.

        **Limitaciones:** UCDP registra eventos letales reportados; los datos Candidate son preliminares; una regresion lineal simple no captura causalidad ni cambios estrategicos en cada conflicto.

        **Siguiente paso:** usar la app de Streamlit para que el publico filtre paises, anos y tipos de violencia durante la presentacion.
        """
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
print(NOTEBOOK)
