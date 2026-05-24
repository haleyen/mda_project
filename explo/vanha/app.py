from shiny import App, ui, reactive
from shinywidgets import output_widget, render_widget
import pandas as pd
import plotly.express as px

# 1. Load data

Data_path = r"D:\KUL MSDS 1\mda_project\data\traffic\traffic_q4_2025_shiny.csv"

df = pd.read_csv(Data_path)

# Basic cleaning
df["municipality"] = df["municipality"].fillna("Unknown")
df["direction_name"] = df["direction_name"].fillna("Unknown direction")
df["site_name"] = df["site_name"].fillna("Unknown site")

df["month"] = df["month"].astype(int)
df["hour"] = df["hour"].astype(int)
df["total_traffic"] = pd.to_numeric(df["total_traffic"], errors="coerce").fillna(0)

# Keep only rows with coordinates
df_map_base = df.dropna(subset=["longitude", "latitude"]).copy()

month_labels = {
    10: "October 2025",
    11: "November 2025",
    12: "December 2025",
}

municipality_choices = ["All"] + sorted(df_map_base["municipality"].dropna().unique().tolist())
direction_choices = ["All"] + sorted(df_map_base["richting"].dropna().unique().tolist())

# 2. UI

app_ui = ui.page_fluid(
    ui.h2("🚲 Cycling Traffic Explorer - Q4 2025"),
    ui.p("Explore cyclist traffic intensity by month, hour, municipality, and direction."),

    ui.layout_sidebar(
        ui.sidebar(
            ui.input_select(
                "month",
                "Month",
                choices={str(k): v for k, v in month_labels.items()},
                selected="10",
            ),

            ui.input_slider(
                "hour_range",
                "Hour range",
                min=0,
                max=23,
                value=(0, 23),
                step=1,
            ),

            ui.input_select(
                "direction",
                "Direction",
                choices=direction_choices,
                selected="All",
            ),

            ui.input_select(
                "municipality",
                "Municipality",
                choices=municipality_choices,
                selected="All",
            ),

            ui.input_select(
                "day_type",
                "Day type",
                choices=["All", "Weekday", "Weekend"],
                selected="All",
            ),
        ),

        ui.card(
            ui.card_header("Traffic intensity by counting site"),
            output_widget("traffic_map"),
        ),
    ),
)

# 3. Server

def server(input, output, session):

    @reactive.calc
    def filtered_data():
        data = df_map_base.copy()

        selected_month = int(input.month())
        hour_min, hour_max = input.hour_range()

        data = data[data["month"] == selected_month]
        data = data[(data["hour"] >= hour_min) & (data["hour"] <= hour_max)]

        if input.direction() != "All":
            data = data[data["richting"] == input.direction()]

        if input.municipality() != "All":
            data = data[data["municipality"] == input.municipality()]

        if input.day_type() == "Weekday":
            data = data[data["is_weekend"] == False]
        elif input.day_type() == "Weekend":
            data = data[data["is_weekend"] == True]

        return data


    @output
    @render_widget
    def traffic_map():
        data = filtered_data()

        map_df = (
            data.groupby(
                [
                    "siteid",
                    "site_name",
                    "municipality",
                    "longitude",
                    "latitude",
                ],
                as_index=False,
            )
            .agg(total_traffic=("total_traffic", "sum"))
        )

        if map_df.empty:
            fig = px.scatter_mapbox()
            fig.update_layout(
                title="No data available for selected filters",
                mapbox_style="open-street-map",
                height=700,
            )
            return fig

        fig = px.scatter_mapbox(
            map_df,
            lat="latitude",
            lon="longitude",
            size="total_traffic",
            color="total_traffic",
            hover_name="site_name",
            hover_data={
                "municipality": True,
                "total_traffic": ":,.0f",
                "latitude": False,
                "longitude": False,
            },
            zoom=8,
            height=700,
            title="Cycling traffic by counting site",
            mapbox_style="open-street-map",
        )

        fig.update_layout(
            margin={"r": 0, "t": 40, "l": 0, "b": 0}
        )

        return fig


app = App(app_ui, server)