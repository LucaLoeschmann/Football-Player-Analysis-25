import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity

# ✅ MUST be the first Streamlit command
st.set_page_config(
    layout="wide",
    page_title="Football Player Analysis Tool 24-25",
    page_icon="⚽",
)

# ----------------------------
# Paths
# ----------------------------
data_dir = "data/"

combined_file_path = data_dir + "combined_df.parquet"
goalkeeper_combined_file_path = data_dir + "goalkeeper_combined_df.parquet"
outfield_file_path = data_dir + "outfield_df.parquet"
goalkeeper_file_path = data_dir + "goalkeepers_df.parquet"

# Radar preset files (written by your notebook script)
RADAR_PRESET_FILES = {
    "Outfield Players": {
        "generic": data_dir + "outfield_radar_percentiles__generic.parquet",
        "defender": data_dir + "outfield_radar_percentiles__defender.parquet",
        "midfielder": data_dir + "outfield_radar_percentiles__midfielder.parquet",
        "attacker": data_dir + "outfield_radar_percentiles__attacker.parquet",
    },
    "Goalkeepers": {
        "generic": data_dir + "goalkeeper_radar_percentiles__generic.parquet",
    },
}

# ✅ UI labels for presets (internal keys stay stable)
RADAR_PRESET_LABELS_OUTFIELD = {
    "generic": "Overview",
    "defender": "Defender Profile",
    "midfielder": "Midfielder Profile",
    "attacker": "Attacker Profile",
}
RADAR_PRESET_LABELS_GOALKEEPER = {
    "generic": "Overview",
}

POS_TOKEN_ORDER = {"DF": 0, "MF": 1, "FW": 2}

def primary_pos(pos: str) -> str:
    """Map combos like 'DF,FW' / 'FW DF' to a primary bucket: DF / MF / FW."""
    if pd.isna(pos):
        return ""

    s = str(pos).upper().replace("/", ",").replace(" ", "")
    parts = [p for p in s.split(",") if p]
    parts = [p for p in parts if p in POS_TOKEN_ORDER]

    if not parts:
        return ""

    return min(parts, key=lambda x: POS_TOKEN_ORDER[x])

# ----------------------------
# Data Loading
# ----------------------------
@st.cache_data(ttl=3600)
def load_all_data():
    combined_df = pd.read_parquet(combined_file_path, engine="fastparquet")
    goalkeeper_combined_df = pd.read_parquet(goalkeeper_combined_file_path, engine="fastparquet")
    outfield_df = pd.read_parquet(outfield_file_path, engine="fastparquet")
    goalkeeper_df = pd.read_parquet(goalkeeper_file_path, engine="fastparquet")
    return combined_df, goalkeeper_combined_df, outfield_df, goalkeeper_df


@st.cache_data(ttl=3600)
def load_radar_percentiles(player_type: str, preset_name: str) -> pd.DataFrame:
    # ✅ safer: don't crash on missing preset/file
    try:
        path = RADAR_PRESET_FILES[player_type][preset_name]
        return pd.read_parquet(path, engine="fastparquet")
    except Exception as e:
        st.error(f"Could not load radar preset '{preset_name}' for '{player_type}'.\n\n{e}")
        return pd.DataFrame()


# ----------------------------
# Helper Functions
# ----------------------------
@st.cache_data
def consolidate_player_data(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "Age"]
    consolidated = (
        df.groupby("Player Name", as_index=False)
        .agg(
            {
                "Position": "first",
                "Team": lambda x: ", ".join(pd.Series(x).dropna().unique()),
                "Competition": lambda x: ", ".join(pd.Series(x).dropna().unique()),
                "Age": "first",
                "Nationality": "first",
                **{c: "sum" for c in num_cols},
            }
        )
    )
    return consolidated


@st.cache_data
def apply_filters(
    df: pd.DataFrame,
    apply_age_filter: bool,
    age_range: tuple,
    apply_nationality_filter: bool,
    nationalities: list,
    apply_competition_filter: bool,
    competition: list,
    apply_position_filter: bool,
    positions: list,
) -> pd.DataFrame:
    out = df.copy()

    if apply_age_filter:
        out = out[(out["Age"] >= age_range[0]) & (out["Age"] <= age_range[1])]

    if apply_nationality_filter and nationalities:
        out = out[out["Nationality"].isin(nationalities)]

    if apply_position_filter and positions:
        out = out[out["Position"].isin(positions)]

    if apply_competition_filter and competition:

        def comp_match(val):
            if pd.isna(val):
                return False
            if isinstance(val, list):
                comps = [str(x).strip() for x in val]
            else:
                comps = [c.strip() for c in str(val).split(",")]
            return any(league in comps for league in competition)

        out = out[out["Competition"].apply(comp_match)]

    return out


@st.cache_data
def aggregate_player_data(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "Age"]
    agg = (
        df.groupby("Player Name", as_index=False)
        .agg(
            {
                "Position": "first",
                "Team": lambda x: ", ".join(pd.Series(x).dropna().unique()),
                "Competition": lambda x: ", ".join(pd.Series(x).dropna().unique()),
                "Age": "first",
                "Nationality": "first",
                **{c: "sum" for c in num_cols},
            }
        )
    )
    return agg


RADAR_PRESETS_OUTFIELD = {
    "generic": [
        "SCA", "npxG",
        "Progressive Passes", "Progressive Carries",
        "Passes Attempted",
        "Touches Attacking Penalty Area",
        "Progressive Passes Received",
        "Passes into Penalty Area",
        "Tackles", "Interceptions", "Ball Recoveries",
        "Aerials Won",
    ],
    "defender": [
        "Tackles", "Interceptions", "Clearances", "Ball Recoveries",
        "Touches Defensive 3rd",
        "Aerials Won",
        "Passes Attempted", "Progressive Passes", "Progressive Passing Distance",
        "Progressive Carries",
        "Dribblers Challenged",
    ],
    "midfielder": [
        "Progressive Passes", "Progressive Carries", "Progressive Passing Distance",
        "Passes Attempted",
        "Touches Midfield 3rd", "Touches Attacking 3rd",
        "SCA", "Key Passes",
        "Tackles", "Interceptions", "Ball Recoveries",
        "Progressive Passes Received",
    ],
    "attacker": [
        "npxG", "Shots",
        "SCA", "xAG",
        "Touches Attacking Penalty Area",
        "Progressive Passes Received",
        "Take Ons Attempted", "Successful Take-Ons",
        "Progressive Carries",
        "Passes into Penalty Area",
        "Aerials Won",
        "Tackles",
    ],
}

RADAR_PRESETS_GOALKEEPER = {
    "generic": [
        "PSxG-GA",
        "Save Percentage",
        "Crosses Stopped Percentage",
        "Defensive Action Outside Penalty Area",
        "Average Distance of Defensive Action",
        "Passes Launched Percentage",
        "Launches Completion Percentage",
        "Average Pass Length",
    ]
}


@st.cache_data
def get_similar_players_cosine(selected_player, df, n_top=10):
    df = df.copy().set_index("Player Name")

    features = df.select_dtypes(include=[np.number]).copy()
    if "Age" in features.columns:
        features = features.drop(columns=["Age"])

    if selected_player not in df.index:
        raise ValueError(f"{selected_player} not found in the DataFrame")

    selected_player_features = features.loc[selected_player].values.reshape(1, -1)
    similarities = cosine_similarity(selected_player_features, features)[0]

    df["Similarity"] = similarities
    similar_players = df.drop(selected_player)
    similar_players = (
        similar_players[["Similarity"]]
        .sort_values(by="Similarity", ascending=False)
        .head(n_top)
        .reset_index()
    )
    similar_players["Rank"] = range(1, len(similar_players) + 1)

    df = df.reset_index()
    relevant_columns = ["Player Name", "Position", "Team", "Competition", "Age", "Nationality"]
    similar_players = similar_players.merge(df[relevant_columns], on="Player Name")

    return similar_players[["Rank", "Player Name", "Position", "Team", "Competition", "Age", "Nationality"]]


def wrap_theta_label(s: str, max_len: int = 14) -> str:
    """
    Word-wrap for polar axis labels.
    We'll later convert '\n' -> '<br>' for ticktext (Plotly-safe).
    """
    words = str(s).split(" ")
    lines, cur = [], ""
    for w in words:
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= max_len:
            cur += " " + w
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def create_radar_plot_from_percentiles(df_percentiles, player_names, radar_columns, title_suffix: str = ""):
    colorway = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442"]
    fig = go.Figure()

    theta_raw = list(radar_columns)
    theta_closed = theta_raw + [theta_raw[0]]

    ticktext_wrapped = [wrap_theta_label(c, max_len=18).replace("\n", "<br>") for c in theta_raw]

    for i, player_name in enumerate(player_names):
        row_df = df_percentiles[df_percentiles["Player Name"] == player_name]
        if row_df.empty:  # ✅ avoid IndexError
            continue
        row = row_df.iloc[0]
        values = list(row[radar_columns].values)

        r = values + [values[0]]

        color = colorway[i % len(colorway)]
        label = f"{player_name} ({row['Position']}, {row['Team']})"

        # 1) FILL (no hover)
        fig.add_trace(
            go.Scatterpolar(
                r=r,
                theta=theta_closed,
                fill="toself",
                fillcolor=color,
                opacity=0.30,
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        # 2) LINE + MARKERS (hover)
        fig.add_trace(
            go.Scatterpolar(
                r=r,
                theta=theta_closed,
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=6),
                name=label,
                hoveron="points",
                customdata=theta_closed,
                hovertemplate="<b>%{customdata}</b><br>%{r:.0f}th percentile<extra></extra>",
            )
        )

    title = "Radar Chart Comparison (Percentile Rankings)"
    if title_suffix:
        title = f"{title_suffix} – Percentile Radar"

    fig.update_layout(
        title=title,
        hovermode="closest",
        height=500,
        margin=dict(l=140, r=140, t=70, b=70),
        polar=dict(
            radialaxis=dict(
                range=[0, 100],
                tickvals=[0, 20, 40, 60, 80],
                ticktext=["0%", "20%", "40%", "60%", "80%"],
            ),
            angularaxis=dict(
                tickmode="array",
                tickvals=theta_raw,
                ticktext=ticktext_wrapped,
                tickfont=dict(size=11),
            ),
        ),
        showlegend=True,
    )

    st.plotly_chart(fig, use_container_width=True)


@st.cache_data
def build_comparison_table(df_input, players, stats, per90):
    df_player = df_input[df_input["Player Name"].isin(players)].copy()
    if len(df_player) == 0 or len(stats) == 0:
        return pd.DataFrame()

    if per90 and "90s" not in df_player.columns:
        return pd.DataFrame()

    df_show = df_player[
        ["Player Name", "Team", "Position"]
        + (["90s"] if "90s" in df_player.columns else [])
        + stats
    ].copy()

    if per90:
        for col in stats:
            if "Percentage" not in col and col != "90s":
                df_show[col] = np.where(df_show["90s"] > 0, df_show[col] / df_show["90s"], np.nan)

    return df_show.set_index("Player Name")[stats].T


# ----------------------------
# Tabs / Pages
# ----------------------------
def page_intro():
    # ----------------------------
    # Centered title
    # ----------------------------
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 0.25rem;'>Football Player Analysis Tool 24–25</h1>",
        unsafe_allow_html=True,
    )

    # ----------------------------
    # Main intro text (clean, high-level)
    # ----------------------------
    st.markdown(
        """
        <div style="text-align: center; max-width: 920px; margin: 0.25rem auto 0.6rem auto; line-height: 1.55;">
            This app analyses footballers through <b>playing style and tactical role</b>.<br><br>
            The focus is on <b>player-type comparison and replacement scouting</b>:
            identifying players who behave similarly on the pitch, even when their output
            differs due to team context such as system, possession share or teammates.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ----------------------------
    # Similarity explanation (same style, single line)
    # ----------------------------
    st.markdown(
        """
        <div style="text-align: center; max-width: 920px; margin: 0.1rem auto 0.7rem auto; line-height: 1.55;">
            Similarity is built on <b>per-90 normalised, role-defining metrics</b> —
            ball progression · chance involvement · defensive activity
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ----------------------------
    # Thin limitations strip (blue, centered)
    # ----------------------------
    left, mid, right = st.columns([1, 6, 1])
    with mid:
        st.info(
            "**LIMITATIONS:** "
            "**Top 5 European leagues · League matches only · "
            "Similarity model requires at least 450 minutes played**"
        )

    # ----------------------------
    # Capability overview
    # ----------------------------
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Similar Players**")
        st.caption("Identify players with comparable tactical behaviour and role profile.")

    with c2:
        st.markdown("**Radar Profiles**")
        st.caption("Visualise percentile-based profiles within positional groups.")

    with c3:
        st.markdown("**Head-to-Head**")
        st.caption("Compare selected stats (raw or per 90) to assess output and fit.")

def page_similar_players(combined_df, goalkeeper_combined_df):
    st.header("🔎 Similar Players")

    left, right = st.columns([1, 1])

    with left:
        recommendation_type = st.selectbox(
            "Choose the type of recommendation:",
            ["Outfield Players", "Goalkeepers"],
            key="rec_type",
        )

        consolidated_df = (
            consolidate_player_data(combined_df)
            if recommendation_type == "Outfield Players"
            else consolidate_player_data(goalkeeper_combined_df)
        )
        
        # add simplified position group for filtering (DF/MF/FW)
        consolidated_df = consolidated_df.copy()
        consolidated_df["PosGroup"] = consolidated_df["Position"].apply(primary_pos)
        consolidated_df = consolidated_df[consolidated_df["PosGroup"] != ""]

        player_info = (
            consolidated_df["Player Name"].astype(str)
            + " ("
            + consolidated_df["Team"].astype(str)
            + ", "
            + consolidated_df["Position"].astype(str)
            + ")"
        ).tolist()
        player_name_map = dict(zip(player_info, consolidated_df["Player Name"].values))

        selected_players_info = st.selectbox(
            "Select a player to find similar players:",
            player_info,
            key="rec_player",
        )
        selected_player = player_name_map[selected_players_info]

        n_top = st.number_input(
            "Number of similar players to display:",
            min_value=1,
            max_value=50,
            value=10,
            key="rec_topn",
        )

        with st.expander("Filters", expanded=True):
            apply_age_filter = st.checkbox("Filter by Age Range", key="rec_age_chk")
            apply_nationality_filter = st.checkbox("Filter by Nationality", key="rec_nat_chk")
            apply_competition_filter = st.checkbox("Filter by Competition", key="rec_comp_chk")
            apply_position_filter = st.checkbox("Filter by Position", key="rec_pos_chk")

            age_range = (16, 50)
            nationalities = []
            competition = []
            positions = []

            if apply_age_filter:
                age_range = st.slider(
                    "Select age range of similar players:",
                    min_value=16,
                    max_value=50,
                    value=(18, 30),
                    key="rec_age_rng",
                )

            if apply_nationality_filter:
                nationalities = st.multiselect(
                    "Select nationality of similar players:",
                    options=sorted(consolidated_df["Nationality"].dropna().unique()),
                    key="rec_nat_sel",
                )

            if apply_competition_filter:
                all_competitions = set()
                for comps in consolidated_df["Competition"].dropna():
                    all_competitions.update(str(comps).split(", "))
                competition = st.multiselect(
                    "Select Competition of similar players:",
                    sorted(all_competitions),
                    key="rec_comp_sel",
                )

            if apply_position_filter:
                ordered = ["DF", "MF", "FW"]
                available = [p for p in ordered if p in consolidated_df["PosGroup"].unique()]
                positions = st.multiselect(
                    "Select position group(s):",
                    options=available,
                    key="rec_pos_sel",
                )


        run_btn = st.button("Find Similar Players", key="rec_btn")

    with right:
        st.subheader("Results")

        if run_btn:
            filtered_df = apply_filters(
                consolidated_df,
                apply_age_filter,
                age_range,
                apply_nationality_filter,
                nationalities,
                apply_competition_filter,
                competition,
                apply_position_filter,
                positions,
            )

            if apply_position_filter and positions:
                filtered_df = filtered_df.copy()
                filtered_df["PosGroup"] = filtered_df["Position"].apply(primary_pos)
                filtered_df = filtered_df[filtered_df["PosGroup"].isin(positions)]

            if selected_player not in filtered_df["Player Name"].values:
                selected_player_data = consolidated_df[consolidated_df["Player Name"] == selected_player]
                filtered_df = pd.concat([filtered_df, selected_player_data], ignore_index=True)

            similar_players = get_similar_players_cosine(selected_player, filtered_df, n_top=n_top)
            st.dataframe(similar_players, hide_index=True, use_container_width=True)
        else:
            st.info("Select options on the left, then click **Find Similar Players**.")


def page_radar():
    st.header("📊 Radar Comparison")

    left, right = st.columns([1, 1])

    with left:
        player_type = st.radio(
            "Select player type:",
            ["Outfield Players", "Goalkeepers"],
            key="rad_type",
            horizontal=True,
        )

        # ✅ Outfield: choose preset | ✅ Goalkeepers: no preset selector needed
        if player_type == "Outfield Players":
            preset_options = ["generic", "defender", "midfielder", "attacker"]
            presets_dict = RADAR_PRESETS_OUTFIELD
            labels = RADAR_PRESET_LABELS_OUTFIELD

            preset_name = st.selectbox(
                "Radar profile:",
                preset_options,
                format_func=lambda x: labels.get(x, x),
                key="rad_preset",
            )
            title_suffix = labels.get(preset_name, preset_name)

        else:
            preset_name = "generic"
            presets_dict = RADAR_PRESETS_GOALKEEPER
            labels = RADAR_PRESET_LABELS_GOALKEEPER
            title_suffix = "Goalkeeper Overview"
            st.caption("Radar profile: **Overview**")

        df_radar = load_radar_percentiles(player_type, preset_name)
        if df_radar.empty:
            st.stop()

        radar_columns = presets_dict[preset_name]

        radar_player_info = (
            df_radar["Player Name"].astype(str)
            + " ("
            + df_radar.get("Team", "").astype(str)
            + ", "
            + df_radar.get("Position", "").astype(str)
            + ")"
        ).tolist()

        radar_name_map = dict(zip(radar_player_info, df_radar["Player Name"].values))

        selected_radar_info = st.multiselect(
            "Select players to compare (max 5 recommended):",
            radar_player_info,
            default=[radar_player_info[0]] if len(radar_player_info) > 0 else [],
            key="rad_players",
        )
        selected_radar_players = [radar_name_map[x] for x in selected_radar_info]

        rad_btn = st.button("Generate Radar Plot", key="rad_btn")

    with right:
        st.subheader("Radar")

        if not rad_btn:
            st.info("Select players (and a profile for Outfield), then click **Generate Radar Plot**.")
            return

        if len(selected_radar_players) == 0:
            st.warning("Select at least one player.")
            return

        # ✅ enforce max 5 (since you recommend it)
        if len(selected_radar_players) > 5:
            st.warning("Please select no more than 5 players.")
            return

        create_radar_plot_from_percentiles(
            df_radar,
            selected_radar_players,
            radar_columns,
            title_suffix=title_suffix,
        )


def page_head_to_head(outfield_df, goalkeeper_df):
    st.header("⚔️ Head-to-Head Player Comparison")

    left, right = st.columns([1, 1])

    with left:
        player_type = st.radio(
            "Select Player Type",
            ["Outfield Players", "Goalkeepers"],
            key="h2h_type",
            horizontal=True,
        )
        df_raw = outfield_df if player_type == "Outfield Players" else goalkeeper_df
        df_agg = aggregate_player_data(df_raw)

        df_agg["Player Info"] = (
            df_agg["Player Name"].astype(str)
            + " ("
            + df_agg["Team"].astype(str)
            + ", "
            + df_agg["Position"].astype(str)
            + ")"
        )

        player_options = df_agg["Player Info"].tolist()
        p1 = st.selectbox("Select Player 1", player_options, key="h2h_p1")
        p2 = st.selectbox("Select Player 2", player_options, key="h2h_p2")

        selected_players_info = [p1, p2]
        with st.expander("➕ Add More Players"):
            for i in range(3, 6):
                p = st.selectbox(f"Select Player {i}", ["None"] + player_options, key=f"h2h_p{i}")
                if p != "None":
                    selected_players_info.append(p)

        name_map = dict(zip(df_agg["Player Info"], df_agg["Player Name"]))
        selected_players = [name_map[x] for x in selected_players_info]

        excluded_columns = [
            "Player Name",
            "Player Info",
            "Team",
            "Position",
            "Nationality",
            "Competition",
            "Year Born",
            "Age",
        ]
        stat_columns = [c for c in df_agg.columns if c not in excluded_columns]

        selected_stats = st.multiselect("Select Stats to Compare", stat_columns, key="h2h_stats")
        stat_type = st.radio("Select Data Type", ["Raw Stats", "Per 90 Minutes"], key="h2h_dtype", horizontal=True)

        run_h2h = st.button("Build Comparison Table", key="h2h_btn")

    with right:
        st.subheader("Table")
        if not run_h2h:
            st.info("Choose players + stats on the left, then click **Build Comparison Table**.")
            return

        comparison_df = build_comparison_table(
            df_agg,
            selected_players,
            selected_stats,
            per90=(stat_type == "Per 90 Minutes"),
        )

        if comparison_df.empty:
            st.info("Select stats to display the comparison table.")
            return

        negative_stats = [
            "Yellow Cards",
            "Red Cards",
            "Errors",
            "Miscontrols",
            "Dispossessed",
            "Second Yellow Cards",
            "Fouls Commited",
            "Offsides",
            "Penalty Kicks Conceded",
            "Own Goals",
            "Aerials Lost",
            "Goals Against",
            "Goals Against/90",
            "Penalty Kicks Allowed",
            "Penalty Kicks Missed",
            "Free Kick Goals Against",
            "Corner Kick Goals Against",
            "Own Goals Scored Against Goalkeeper",
            "Shots on Target Against",
        ]

        def format_number(x):
            if pd.isna(x):
                return ""
            if isinstance(x, (int, np.integer)):
                return f"{x}"
            if isinstance(x, (float, np.floating)):
                return f"{x:.2f}" if stat_type == "Per 90 Minutes" else f"{x:.1f}"
            return str(x)

        comparison_fmt = comparison_df.copy().applymap(format_number)

        def highlight_best(row_display):
            stat_name = row_display.name
            is_negative = stat_name in negative_stats

            row_num = comparison_df.loc[stat_name]
            if row_num.nunique(dropna=True) <= 1:
                return [""] * len(row_display)

            best_val = row_num.min(skipna=True) if is_negative else row_num.max(skipna=True)

            styles = []
            for v in row_num:
                if pd.isna(v):
                    styles.append("")
                elif (not is_negative) and v == best_val:
                    styles.append("background-color: lightgreen")
                elif is_negative and v == best_val:
                    styles.append("background-color: lightcoral")
                else:
                    styles.append("")
            return styles

        styled_df = comparison_fmt.style.apply(highlight_best, axis=1)
        st.dataframe(styled_df, use_container_width=True)


# ----------------------------
# Main
# ----------------------------
def main():
    combined_df, goalkeeper_combined_df, outfield_df, goalkeeper_df = load_all_data()

    page_intro()
    st.divider()

    tab1, tab2, tab3 = st.tabs(["🔎 Similar Players", "📊 Radar", "⚔️ Head-to-Head"])

    with tab1:
        page_similar_players(combined_df, goalkeeper_combined_df)

    with tab2:
        page_radar()

    with tab3:
        page_head_to_head(outfield_df, goalkeeper_df)


if __name__ == "__main__":
    main()
