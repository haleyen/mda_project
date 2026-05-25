import os
import pandas as pd
import plotly.express as px

from shiny import App, ui, reactive
from shinywidgets import output_widget, render_widget


# 0. config/ file paths


BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()

PATH_DATA = os.path.join(BASE_DIR, "data")
PATH_TRAFFIC = os.path.join(PATH_DATA, "traffic")

FILEPATH_RICH = os.path.join(PATH_DATA, "richtingen.csv")
FILEPATH_SITE = os.path.join(PATH_DATA, "sites.csv")

FILEPATH_TRAFFIC_OCT = os.path.join(PATH_TRAFFIC, "data-2025-10.csv")
FILEPATH_TRAFFIC_NOV = os.path.join(PATH_TRAFFIC, "data-2025-11.csv")
FILEPATH_TRAFFIC_DEC = os.path.join(PATH_TRAFFIC, "data-2025-12.csv")

FILEPATH_TRAFFIC_AGG = os.path.join(PATH_TRAFFIC, "traffic_q4_2025_aggregated.csv")
FILEPATH_TRAFFIC_SHINY = os.path.join(PATH_TRAFFIC, "traffic_q4_2025_shiny.csv")

TRAFFIC_COLS = ["siteid", "richting", "type", "van", "tot", "aantal"]

RICH_COLS = ["siteid", "richting", "naam"]

SITE_COLS = [
    "siteid",
    "sitenr",
    "longitude",
    "latitude",
    "site_name",
    "domein",
    "wegnr",
    "district",
    "municipality",
    "interval",
    "install_date",
]


# 1. prepare aggregated CSV

def create_aggregated_traffic_csv():
    """
    Reads raw Q4 2025 traffic files, cleans them, aggregates them to hourly level,
    and exports traffic_q4_2025_aggregated.csv.
    """

    traffic_files = [
        FILEPATH_TRAFFIC_OCT,
        FILEPATH_TRAFFIC_NOV,
        FILEPATH_TRAFFIC_DEC,
    ]

    all_agg = []

    for file in traffic_files:
        print(f"\nProcessing raw traffic file: {os.path.basename(file)}")

        df = pd.read_csv(
            file,
            header=None,
            names=TRAFFIC_COLS,
        )

        print("Raw shape:", df.shape)

        # Convert timestamps
        df["van"] = pd.to_datetime(df["van"], errors="coerce")
        df["tot"] = pd.to_datetime(df["tot"], errors="coerce")

        # Remove invalid timestamps
        df = df[df["van"].notna()]

        # Convert count to numeric
        df["aantal"] = pd.to_numeric(df["aantal"], errors="coerce")
        df = df[df["aantal"].notna()]

        # Keep cyclists only
        df = df[df["type"].astype(str).str.upper().eq("FIETSERS")]

        # Keep only Q4 2025
        df = df[(df["van"] >= "2025-10-01") & (df["van"] < "2026-01-01")]

        # Create time features
        df["year"] = df["van"].dt.year
        df["month"] = df["van"].dt.month
        df["day"] = df["van"].dt.day
        df["hour"] = df["van"].dt.hour
        df["weekday"] = df["van"].dt.day_name()
        df["is_weekend"] = df["van"].dt.dayofweek >= 5

        # Aggregate to hourly level
        agg = (
            df.groupby(
                [
                    "siteid",
                    "richting",
                    "year",
                    "month",
                    "day",
                    "hour",
                    "weekday",
                    "is_weekend",
                ],
                as_index=False,
            )["aantal"]
            .sum()
            .rename(columns={"aantal": "total_traffic"})
        )

        print("Aggregated shape:", agg.shape)
        all_agg.append(agg)

    traffic_agg = pd.concat(all_agg, ignore_index=True)

    # Re-aggregate in case duplicate groups exist
    traffic_agg = (
        traffic_agg.groupby(
            [
                "siteid",
                "richting",
                "year",
                "month",
                "day",
                "hour",
                "weekday",
                "is_weekend",
            ],
            as_index=False,
        )["total_traffic"]
        .sum()
    )

    traffic_agg.to_csv(FILEPATH_TRAFFIC_AGG, index=False)

    print("\nAggregated traffic CSV saved:")
    print(FILEPATH_TRAFFIC_AGG)
    print("Rows:", len(traffic_agg))

    return traffic_agg



# 2. Create shiny ready csv


def create_shiny_ready_csv():
    """
    Creates the aggregated CSV and merges it with sites and directions metadata.
    Exports traffic_q4_2025_shiny.csv.
    """

    traffic_agg = create_aggregated_traffic_csv()

    print("\nLoading metadata files...")

    df_rich = pd.read_csv(
        FILEPATH_RICH,
        header=None,
        names=RICH_COLS,
    )

    df_sites = pd.read_csv(
        FILEPATH_SITE,
        header=None,
        names=SITE_COLS,
    )

    # Remove problematic direction row
    df_rich = df_rich.query('richting != "IN/OUT"').copy()

    # Rename direction name
    df_rich = df_rich.rename(columns={"naam": "direction_name"})

    # Clean merge keys
    traffic_agg["siteid"] = pd.to_numeric(traffic_agg["siteid"], errors="coerce")
    df_sites["siteid"] = pd.to_numeric(df_sites["siteid"], errors="coerce")
    df_rich["siteid"] = pd.to_numeric(df_rich["siteid"], errors="coerce")

    traffic_agg["richting"] = traffic_agg["richting"].astype(str).str.strip().str.upper()
    df_rich["richting"] = df_rich["richting"].astype(str).str.strip().str.upper()

    # Merge traffic with site metadata
    shiny_df = traffic_agg.merge(
        df_sites,
        on="siteid",
        how="left",
    )

    # Merge traffic with direction metadata
    shiny_df = shiny_df.merge(
        df_rich,
        on=["siteid", "richting"],
        how="left",
    )

    print("\nFinal Shiny-ready dataset shape:", shiny_df.shape)

    print("\nMissing values after merge:")
    print(
        shiny_df[
            ["longitude", "latitude", "municipality", "site_name", "direction_name"]
        ].isna().sum()
    )

    shiny_df.to_csv(FILEPATH_TRAFFIC_SHINY, index=False)

    print("\nShiny-ready CSV saved:")
    print(FILEPATH_TRAFFIC_SHINY)

    return shiny_df


# 3. run preparation before starting app


df = create_shiny_ready_csv()


# 4. basic cleaning for app

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


# 5. UI

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


# 6. Server


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