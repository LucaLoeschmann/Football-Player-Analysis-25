import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

# ✅ MUST be the first Streamlit command
st.set_page_config(
    layout="wide",
    page_title="Football Player Analysis Tool 24-25",
    page_icon="⚽",
)

# ----------------------------
# Paths (Streamlit Cloud ready)
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
data_dir = BASE_DIR / "data/"

combined_file_path = data_dir / "combined_df.parquet"
goalkeeper_combined_file_path = data_dir / "goalkeeper_combined_df.parquet"
outfield_file_path = data_dir / "outfield_df.parquet"
goalkeeper_file_path = data_dir / "goalkeepers_df.parquet"

# ✅ Reference files = benchmark distributions for DF/MF/FW etc.
RADAR_REFERENCE_FILES = {
    "Outfield Players": {
        "generic": data_dir / "outfield_radar_reference__generic.parquet",
        "defender": data_dir / "outfield_radar_reference__defender.parquet",
        "midfielder": data_dir / "outfield_radar_reference__midfielder.parquet",
        "attacker": data_dir / "outfield_radar_reference__attacker.parquet",
    },
    "Goalkeepers": {
        "generic": data_dir / "goalkeeper_radar_reference__generic.parquet",
    },
}

# ----------------------------
# Per90 exceptions (do NOT divide)
# ----------------------------
NON_PER90_OUTFIELD = [
    "Pass Completion Percentage",
    "Long Pass Completion Percentage",
    "Short Pass Completion Percentage",
    "Medium Pass Completion Percentage",
    "Shots on Target Percentage",
    "Successful Take-On Percentage",
    "Dribblers Tackled Percentage",
    "Tackled During Take-On Percentage",
    "npxG per Shot",
    "Average Shot Distance",
]

NON_PER90_GOALKEEPER = [
    "Save Percentage",
    "Crosses Stopped Percentage",
    "Passes Launched Percentage",
    "Launches Completion Percentage",
    "Average Distance of Defensive Action",
    "Average Pass Length",
    "PSxG-GA",
]

# ----------------------------
# ✅ Aggregation exceptions (do NOT sum in groupby)
# ----------------------------
# Alles, was "Rate/Percentage/Average/Per Shot" ist, darf beim Aggregieren nicht summiert werden.
# Wir halten diese Spalten im aggregierten DF via "first".
NON_AGG_SUM_COLS = set(
    NON_PER90_OUTFIELD
    + NON_PER90_GOALKEEPER
    + [
        # GK/Outfield extra rates/averages
        "Clean Sheet Percentage",
        "Penalty Kicks Save Percentage",
        "Goal Kicks Average Length",
        "Post-Shot Expected Goals per Shot on Target",
        # falls in deinen DF vorhanden:
        "Goals per Shot",
        "Goals per Shot on Target",
        "Aerials Win Percentage",
    ]
)

# ----------------------------
# UI labels for presets
# ----------------------------
RADAR_PRESET_LABELS_OUTFIELD = {
    "generic": "Overview",
    "defender": "Defender Profile",
    "midfielder": "Midfielder Profile",
    "attacker": "Attacker Profile",
}
RADAR_PRESET_LABELS_GOALKEEPER = {"generic": "Overview"}

# ----------------------------
# Position parsing / ordering (DF/MF/FW)
# ----------------------------
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
def load_radar_reference(player_type: str, preset_name: str) -> pd.DataFrame:
    path = RADAR_REFERENCE_FILES[player_type][preset_name]
    return pd.read_parquet(path, engine="fastparquet")

# ----------------------------
# ✅ ONE aggregation function for all pages
# ----------------------------
@st.cache_data
def aggregate_players_safely(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregiert pro Player Name.
    - Zählwerte werden summiert (Goals, Shots, Tackles, etc.)
    - Percentages/Averages/Rate-Spalten werden NICHT summiert, sondern via "first" behalten
    """
    df = df.copy()

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # numerische Spalten, die wir summieren dürfen (alles außer Age und NON_AGG_SUM_COLS)
    sum_cols = [c for c in num_cols if (c != "Age") and (c not in NON_AGG_SUM_COLS)]

    # numerische Spalten, die wir NICHT summieren, aber behalten wollen
    keep_first_cols = [c for c in num_cols if c in NON_AGG_SUM_COLS]

    agg_dict = {
        "Position": "first",
        "Team": lambda x: ", ".join(pd.Series(x).dropna().unique()),
        "Competition": lambda x: ", ".join(pd.Series(x).dropna().unique()),
        "Age": "first",
        "Nationality": "first",
        **({"Year Born": "first"} if "Year Born" in df.columns else {}),
        **{c: "sum" for c in sum_cols},
        **{c: "first" for c in keep_first_cols},
    }

    out = df.groupby("Player Name", as_index=False).agg(agg_dict)
    return out

# ----------------------------
# Filters
# ----------------------------
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
    pos_groups: list,
) -> pd.DataFrame:
    out = df.copy()

    if apply_age_filter:
        out = out[(out["Age"] >= age_range[0]) & (out["Age"] <= age_range[1])]

    if apply_nationality_filter and nationalities:
        out = out[out["Nationality"].isin(nationalities)]

    if apply_position_filter and pos_groups and "PosGroup" in out.columns:
        out = out[out["PosGroup"].isin(pos_groups)]

    if apply_competition_filter and competition:
        def comp_match(val):
            if pd.isna(val):
                return False
            comps = [c.strip() for c in str(val).split(",")]
            return any(league in comps for league in competition)

        out = out[out["Competition"].apply(comp_match)]

    return out

# ----------------------------
# Radar presets
# ----------------------------
RADAR_PRESETS_OUTFIELD = {
    "generic": [
        "SCA",
        "npxG",
        "Progressive Passes",
        "Progressive Carries",
        "Passes Attempted",
        "Touches Attacking Penalty Area",
        "Progressive Passes Received",
        "Passes into Penalty Area",
        "Tackles",
        "Interceptions",
        "Ball Recoveries",
        "Aerials Won",
    ],
    "defender": [
        "Tackles",
        "Interceptions",
        "Clearances",
        "Ball Recoveries",
        "Touches Defensive 3rd",
        "Aerials Won",
        "Passes Attempted",
        "Progressive Passes",
        "Progressive Passing Distance",
        "Progressive Carries",
        "Dribblers Challenged",
    ],
    "midfielder": [
        "Progressive Passes",
        "Progressive Carries",
        "Progressive Passing Distance",
        "Passes Attempted",
        "Touches Midfield 3rd",
        "Touches Attacking 3rd",
        "SCA",
        "Key Passes",
        "Tackles",
        "Interceptions",
        "Ball Recoveries",
        "Progressive Passes Received",
    ],
    "attacker": [
        "npxG",
        "Shots",
        "SCA",
        "xAG",
        "Touches Attacking Penalty Area",
        "Progressive Passes Received",
        "Take Ons Attempted",
        "Successful Take-Ons",
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

# ----------------------------
# Similarity
# ----------------------------
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

# ----------------------------
# Radar percentile computation vs reference distribution
# ----------------------------
def _empirical_percentile(value: float, ref_values: np.ndarray) -> float:
    """Percentile = % of reference values <= value (0..100)."""
    ref_values = ref_values[np.isfinite(ref_values)]
    if ref_values.size == 0 or not np.isfinite(value):
        return 0.0
    ref_sorted = np.sort(ref_values)
    idx = np.searchsorted(ref_sorted, value, side="right")
    return round(100.0 * idx / ref_sorted.size, 1)

def compute_percentiles_vs_reference(
    raw_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    player_names: list[str],
    radar_cols: list[str],
    non_per90_cols: list[str],
) -> pd.DataFrame:
    if raw_df.empty or reference_df.empty or len(player_names) == 0:
        return pd.DataFrame()

    meta_cols = ["Player Name", "Team", "Position", "Age", "Nationality", "Competition"]

    sel = raw_df[raw_df["Player Name"].isin(player_names)].copy()
    if sel.empty:
        return pd.DataFrame()

    for c in radar_cols + ["90s"]:
        if c in sel.columns:
            sel[c] = pd.to_numeric(sel[c], errors="coerce")
        if c in reference_df.columns:
            reference_df[c] = pd.to_numeric(reference_df[c], errors="coerce")

    per90_cols = [c for c in radar_cols if c not in non_per90_cols]
    if "90s" in sel.columns:
        sel[per90_cols] = sel[per90_cols].div(sel["90s"].replace(0, np.nan), axis=0)

    out = sel[meta_cols + radar_cols].copy()
    for col in radar_cols:
        ref_vals = reference_df[col].to_numpy(dtype=float, copy=False)
        out[col] = out[col].apply(
            lambda v: _empirical_percentile(float(v) if pd.notna(v) else np.nan, ref_vals)
        )

    return out.drop_duplicates(subset=["Player Name"], keep="first")

# ----------------------------
# Plot helpers
# ----------------------------
def wrap_theta_label(s: str, max_len: int = 14) -> str:
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

def create_radar_plot_from_percentiles(df_percentiles, player_names, radar_columns):
    colorway = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442"]
    fig = go.Figure()

    theta_raw = list(radar_columns)
    theta_closed = theta_raw + [theta_raw[0]]
    ticktext_wrapped = [wrap_theta_label(c, max_len=18).replace("\n", "<br>") for c in theta_raw]

    for i, player_name in enumerate(player_names):
        row_df = df_percentiles[df_percentiles["Player Name"] == player_name]
        if row_df.empty:
            continue

        row = row_df.iloc[0]
        values = list(row[radar_columns].values)
        r = values + [values[0]]

        color = colorway[i % len(colorway)]
        label = f"{player_name} ({row.get('Position','')}, {row.get('Team','')})"

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

    fig.update_layout(
        title_text="",
        hovermode="closest",
        height=680,
        margin=dict(l=140, r=120, t=120, b=70),
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=1.12,
            yanchor="top",
            font=dict(size=13),
        ),
        polar=dict(
            domain=dict(x=[0.04, 0.98], y=[0.00, 0.86]),
            radialaxis=dict(
                range=[0, 100],
                tickvals=[0, 20, 40, 60, 80],
                ticktext=["0%", "20%", "40%", "60%", "80%"],
                tickfont=dict(size=12),
            ),
            angularaxis=dict(
                tickmode="array",
                tickvals=theta_raw,
                ticktext=ticktext_wrapped,
                tickfont=dict(size=12),
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
            if col != "90s" and col not in NON_AGG_SUM_COLS:
                df_show[col] = np.where(df_show["90s"] > 0, df_show[col] / df_show["90s"], np.nan)

    return df_show.set_index("Player Name")[stats].T

# ----------------------------
# Pages
# ----------------------------
def page_intro():
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 0.25rem;'>Football Player Analysis Tool 24–25</h1>",
        unsafe_allow_html=True,
    )

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

    st.markdown(
        """
        <div style="text-align: center; max-width: 920px; margin: 0.1rem auto 0.7rem auto; line-height: 1.55;">
            Similarity is built on <b>per-90 normalised, role-defining metrics</b> —
            ball progression · chance involvement · defensive activity
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "**LIMITATIONS:** **Top 5 European leagues · League matches only · Similarity model requires at least 450 minutes played**"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("**Similar Players**")
        st.caption("Find players with comparable tactical behaviour and role profile.")

    with c2:
        st.markdown("**Radar Profiles**")
        st.caption("Compare percentile-based style profiles across benchmark groups.")

    with c3:
        st.markdown("**Head-to-Head**")
        st.caption("Directly compare selected stats to assess output and fit.")

    with c4:
        st.markdown("**Leaderboards**")
        st.caption("Rank players by individual stats with filters and sorting.")

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
            aggregate_players_safely(combined_df)
            if recommendation_type == "Outfield Players"
            else aggregate_players_safely(goalkeeper_combined_df)
        )

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
            index=None,
            placeholder="Choose a player…",
            key="rec_player",
        )
        selected_player = player_name_map.get(selected_players_info) if selected_players_info else None

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
            apply_position_filter = st.checkbox("Filter by Position Group", key="rec_pos_chk")

            age_range = (16, 50)
            nationalities, competition, pos_groups = [], [], []

            if apply_age_filter:
                age_range = st.slider("Age range:", 16, 50, (18, 30), key="rec_age_rng")

            if apply_nationality_filter:
                nationalities = st.multiselect(
                    "Nationality:",
                    options=sorted(consolidated_df["Nationality"].dropna().unique()),
                    key="rec_nat_sel",
                )

            if apply_competition_filter:
                all_competitions = set()
                for comps in consolidated_df["Competition"].dropna():
                    all_competitions.update(str(comps).split(", "))
                competition = st.multiselect(
                    "Competition:",
                    sorted(all_competitions),
                    key="rec_comp_sel",
                )

            if apply_position_filter:
                ordered = ["DF", "MF", "FW"]
                available = [p for p in ordered if p in consolidated_df["PosGroup"].unique()]
                pos_groups = st.multiselect(
                    "Position group:",
                    options=available,
                    key="rec_pos_sel",
                )

        run_btn = st.button(
            "Find Similar Players",
            key="rec_btn",
            disabled=(selected_player is None),
        )

    with right:
        st.subheader("Results")

        if selected_player is None:
            st.info("Select a player on the left to enable recommendations.")
            return

        if not run_btn:
            st.info("Select options on the left, then click **Find Similar Players**.")
            return

        filtered_df = apply_filters(
            consolidated_df,
            apply_age_filter,
            age_range,
            apply_nationality_filter,
            nationalities,
            apply_competition_filter,
            competition,
            apply_position_filter,
            pos_groups,
        )

        if selected_player not in filtered_df["Player Name"].values:
            selected_player_data = consolidated_df[consolidated_df["Player Name"] == selected_player]
            filtered_df = pd.concat([filtered_df, selected_player_data], ignore_index=True)

        similar_players = get_similar_players_cosine(selected_player, filtered_df, n_top=n_top)

        st.caption("Tip: You can scroll horizontally in the table (Shift + mouse wheel / trackpad).")
        st.dataframe(similar_players, hide_index=True, use_container_width=True)

def page_radar(outfield_df, goalkeeper_df):
    st.header("📊 Radar Comparison")
    left, right = st.columns([1, 1])

    with left:
        player_type = st.radio(
            "Select player type:",
            ["Outfield Players", "Goalkeepers"],
            key="rad_type",
            horizontal=True,
        )

        if player_type == "Outfield Players":
            preset_options = ["generic", "defender", "midfielder", "attacker"]
            presets_dict = RADAR_PRESETS_OUTFIELD
            labels = RADAR_PRESET_LABELS_OUTFIELD
            non_per90 = NON_PER90_OUTFIELD
            df_all = outfield_df
        else:
            preset_options = ["generic"]
            presets_dict = RADAR_PRESETS_GOALKEEPER
            labels = RADAR_PRESET_LABELS_GOALKEEPER
            non_per90 = NON_PER90_GOALKEEPER
            df_all = goalkeeper_df

        preset_name = st.selectbox(
            "Radar profile (benchmark group):",
            preset_options,
            format_func=lambda x: labels.get(x, x),
            key="rad_preset",
        )
        radar_columns = presets_dict[preset_name]

        try:
            df_ref = load_radar_reference(player_type, preset_name)
        except Exception as e:
            st.error(f"Could not load benchmark distribution.\n\n{e}")
            st.stop()

        df_all_small = df_all[["Player Name", "Team", "Position"]].copy()
        df_all_small["Player Info"] = (
            df_all_small["Player Name"].astype(str)
            + " ("
            + df_all_small["Team"].astype(str)
            + ", "
            + df_all_small["Position"].astype(str)
            + ")"
        )
        player_options = sorted(df_all_small["Player Info"].dropna().unique().tolist())
        name_map = dict(zip(df_all_small["Player Info"], df_all_small["Player Name"]))

        selected_radar_info = st.multiselect(
            "Select players to compare (max 5 recommended):",
            player_options,
            default=[],
            key="rad_players",
        )
        selected_players = [name_map[x] for x in selected_radar_info if x in name_map]

        if player_type == "Outfield Players":
            bench_label = {
                "generic": "All outfield",
                "defender": "Defenders (DF)",
                "midfielder": "Midfielders (MF)",
                "attacker": "Forwards (FW)",
            }.get(preset_name, "All outfield")
        else:
            bench_label = "Goalkeepers"

        st.caption(f"Benchmark group: **{bench_label}** (percentiles computed vs this group)")
        rad_btn = st.button("Generate Radar Plot", key="rad_btn", disabled=(len(selected_players) == 0))

    with right:
        if len(selected_players) == 0:
            st.info("Select at least one player to enable the radar.")
            return

        if not rad_btn:
            st.info("Select players + a profile, then click **Generate Radar Plot**.")
            return

        if len(selected_players) > 5:
            st.warning("Please select no more than 5 players.")
            return

        df_pct = compute_percentiles_vs_reference(
            raw_df=df_all,
            reference_df=df_ref,
            player_names=selected_players,
            radar_cols=radar_columns,
            non_per90_cols=non_per90,
        )

        if df_pct.empty:
            st.error("Could not compute radar percentiles for the selected players.")
            return

        create_radar_plot_from_percentiles(df_pct, selected_players, radar_columns)

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
        df_agg = aggregate_players_safely(df_raw)

        df_agg["Player Info"] = (
            df_agg["Player Name"].astype(str)
            + " ("
            + df_agg["Team"].astype(str)
            + ", "
            + df_agg["Position"].astype(str)
            + ")"
        )

        player_options = df_agg["Player Info"].tolist()

        p1 = st.selectbox("Select Player 1", player_options, index=None, placeholder="Choose Player 1…", key="h2h_p1")
        p2 = st.selectbox("Select Player 2", player_options, index=None, placeholder="Choose Player 2…", key="h2h_p2")

        selected_players_info = []
        if p1:
            selected_players_info.append(p1)
        if p2:
            selected_players_info.append(p2)

        with st.expander("➕ Add More Players"):
            for i in range(3, 6):
                p = st.selectbox(f"Select Player {i}", ["None"] + player_options, index=0, key=f"h2h_p{i}")
                if p != "None":
                    selected_players_info.append(p)

        name_map = dict(zip(df_agg["Player Info"], df_agg["Player Name"]))
        selected_players = [name_map[x] for x in selected_players_info if x in name_map]

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

        run_h2h = st.button(
            "Build Comparison Table",
            key="h2h_btn",
            disabled=(p1 is None or p2 is None),
        )

    with right:
        st.subheader("Table")

        if p1 is None or p2 is None:
            st.info("Select Player 1 and Player 2 to enable the table.")
            return

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
            "Second Yellow Card",
            "Fouls Committed",
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

        st.caption("Tip: You can scroll horizontally in the table (Shift + mouse wheel / trackpad).")
        st.dataframe(styled_df, use_container_width=True)

def page_leaderboard(outfield_df, goalkeeper_df):
    st.header("🏆 Stat Leaderboard")

    left, right = st.columns([1, 1.4])

    with left:
        player_type = st.radio(
            "Player type:",
            ["Outfield Players", "Goalkeepers"],
            key="lb_type",
            horizontal=True,
        )

        df_raw = outfield_df if player_type == "Outfield Players" else goalkeeper_df
        df_agg = aggregate_players_safely(df_raw).copy()

        df_agg["PosGroup"] = df_agg["Position"].apply(primary_pos)

        with st.expander("Filters", expanded=True):
            apply_age = st.checkbox("Filter by Age Range", key="lb_age_chk")
            apply_nat = st.checkbox("Filter by Nationality", key="lb_nat_chk")
            apply_comp = st.checkbox("Filter by Competition", key="lb_comp_chk")
            apply_pos = st.checkbox("Filter by Position Group", key="lb_pos_chk")

            # ✅ NEW: minutes/90s filters
            apply_90s_filter = st.checkbox("Filter by Minutes (90s) Range", key="lb_90s_chk")
            qualified_only = st.toggle(
                "Qualified only (min 5 full matches / 5.0 90s)",
                value=False,
                key="lb_qual",
            )

            age_range = (16, 50)
            nationalities, competitions, pos_groups = [], [], []
            min_90s_range = None  # ✅ NEW

            if apply_age:
                age_range = st.slider("Age range:", 16, 50, (18, 30), key="lb_age_rng")

            if apply_nat:
                nationalities = st.multiselect(
                    "Nationality:",
                    sorted(df_agg["Nationality"].dropna().unique()),
                    key="lb_nat_sel",
                )

            if apply_comp:
                all_competitions = set()
                for comps in df_agg["Competition"].dropna():
                    all_competitions.update(str(comps).split(", "))
                competitions = st.multiselect(
                    "Competition:",
                    sorted(all_competitions),
                    key="lb_comp_sel",
                )

            if apply_pos:
                ordered = ["DF", "MF", "FW"]
                available = [p for p in ordered if p in df_agg["PosGroup"].unique()]
                pos_groups = st.multiselect(
                    "Position group:",
                    available,
                    key="lb_pos_sel",
                )

            # ✅ NEW: 90s range slider (dynamic limits)
            if apply_90s_filter:
                if "90s" in df_agg.columns:
                    _s90 = pd.to_numeric(df_agg["90s"], errors="coerce")
                    _max90 = float(_s90.max(skipna=True) if _s90.notna().any() else 0.0)
                    _max90 = max(0.0, _max90)

                    min_90s_range = st.slider(
                        "90s range (full matches):",
                        min_value=0.0,
                        max_value=float(np.ceil(_max90)),
                        value=(0.0, float(np.ceil(_max90))),
                        step=0.5,
                        key="lb_90s_rng",
                    )
                else:
                    st.warning("'90s' column missing – cannot filter by minutes.")
                    apply_90s_filter = False

        excluded = {
            "Player Name", "Team", "Position", "Competition", "Nationality", "Age",
            "Year Born", "PosGroup"
        }
        numeric_stats = [c for c in df_agg.select_dtypes(include=[np.number]).columns if c not in excluded]

        stat = st.selectbox(
            "Rank by stat:",
            options=sorted(numeric_stats),
            index=0 if numeric_stats else None,
            key="lb_stat",
        )

        per90 = st.checkbox("Per 90", value=False, key="lb_per90")
        top_n = st.number_input("Show top number:", min_value=5, max_value=200, value=25, step=5, key="lb_topn")

        sort_dir = st.radio("Sort:", ["High", "Low"], horizontal=True, key="lb_sort_dir")
        ascending = (sort_dir == "Low")

        run = st.button("Build Leaderboard", key="lb_btn")

    with right:
        if not run:
            st.info("Select filters + a stat, then click **Build Leaderboard**.")
            return

        if stat is None:
            st.warning("No stat columns available.")
            return

        df = df_agg.copy()

        df = apply_filters(
            df,
            apply_age_filter=apply_age,
            age_range=age_range,
            apply_nationality_filter=apply_nat,
            nationalities=nationalities,
            apply_competition_filter=apply_comp,
            competition=competitions,
            apply_position_filter=apply_pos,
            pos_groups=pos_groups,
        )

        # ✅ NEW: apply 90s filters (Leaderboard only)
        if "90s" in df.columns:
            df["90s"] = pd.to_numeric(df["90s"], errors="coerce")

            if qualified_only:
                df = df[df["90s"] >= 5.0]

            if apply_90s_filter and min_90s_range is not None:
                lo, hi = min_90s_range
                df = df[(df["90s"] >= lo) & (df["90s"] <= hi)]
        else:
            if qualified_only or apply_90s_filter:
                st.warning("'90s' column missing – minutes filters ignored.")

        # ✅ per90: nur teilen, wenn stat NICHT in NON_AGG_SUM_COLS ist (Percentages/Averages bleiben unberührt)
        if per90:
            if "90s" not in df.columns:
                st.error("Per 90 requested but '90s' column is missing.")
                return
            if stat != "90s" and stat not in NON_AGG_SUM_COLS:
                df[stat] = np.where(df["90s"] > 0, df[stat] / df["90s"], np.nan)

        df[stat] = pd.to_numeric(df[stat], errors="coerce")
        df = df.dropna(subset=[stat]).sort_values(stat, ascending=ascending)

        show_cols = ["Player Name", "Team", "Position", "Competition", "Age", "Nationality", "90s", stat]
        show_cols = [c for c in show_cols if c in df.columns]

        seen = set()
        show_cols = [c for c in show_cols if not (c in seen or seen.add(c))]

        out = df[show_cols].head(int(top_n)).copy()
        out.insert(0, "Rank", range(1, len(out) + 1))

        # ✅ NEW: Anzeige-Formatierung wie bei H2H (minimal, Sorting bleibt korrekt)
        def _fmt(x):
            if pd.isna(x):
                return ""
            if isinstance(x, (int, np.integer)):
                return f"{x}"
            if isinstance(x, (float, np.floating)):
                return f"{x:.2f}" if per90 else f"{x:.1f}"
            return str(x)

        out_disp = out.copy()
        if stat in out_disp.columns:
            out_disp[stat] = out_disp[stat].apply(_fmt)
        if "90s" in out_disp.columns:
            out_disp["90s"] = out_disp["90s"].apply(_fmt)

        st.dataframe(out_disp, hide_index=True, use_container_width=True)

# ----------------------------
# Main
# ----------------------------
def main():
    combined_df, goalkeeper_combined_df, outfield_df, goalkeeper_df = load_all_data()

    page_intro()
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(["🔎 Similar Players", "📊 Radar", "⚔️ Head-to-Head", "🏆 Leaderboard"])

    with tab1:
        page_similar_players(combined_df, goalkeeper_combined_df)

    with tab2:
        page_radar(outfield_df, goalkeeper_df)

    with tab3:
        page_head_to_head(outfield_df, goalkeeper_df)

    with tab4:
        page_leaderboard(outfield_df, goalkeeper_df)

if __name__ == "__main__":
    main()
