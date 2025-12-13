import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import percentileofscore
import fastparquet  


# File paths
data_dir = "data/"
combined_file_path = data_dir + "combined_df.parquet"
outfield_file_path = data_dir + "outfield_df.parquet"
goalkeeper_combined_file_path = data_dir + "goalkeeper_combined_df.parquet"
goalkeeper_file_path = data_dir + "goalkeepers_df.parquet"

# Load datasets - cache the data loading process for all files at once
@st.cache_data(ttl=3600)  # Cache for 1 hour (3600 seconds)
def load_all_data():
    # Read parquet files
    combined_df = pd.read_parquet(combined_file_path, engine="fastparquet")
    goalkeeper_combined_df = pd.read_parquet(goalkeeper_combined_file_path, engine="fastparquet")
    outfield_df = pd.read_parquet(outfield_file_path, engine="fastparquet")
    goalkeeper_df = pd.read_parquet(goalkeeper_file_path, engine="fastparquet")
    
    # Return all DataFrames as a tuple
    return combined_df, goalkeeper_combined_df, outfield_df, goalkeeper_df

# Load all data into variables at once
combined_df, goalkeeper_combined_df, outfield_df, goalkeeper_df = load_all_data()

# Function to return the correct dataframe based on input type
def get_dataframe(type_):
    if type_ == "Outfield Players":
        return combined_df
    elif type_ == "Goalkeepers Combined":
        return goalkeeper_combined_df
    elif type_ == "Outfield Players":
        return outfield_df
    elif type_ == "Goalkeepers":
        return goalkeeper_df
    return None

# Radar chart columns for outfield players and goalkeepers
radar_columns_outfield = [
    "Goals-PK", "npxG", "Shots", "Assists", "xAG", "npxG+xAG", "SCA", 
    "Passes Attempted", "Pass Completion Percentage", "Progressive Passes", 
    "Progressive Carries", "Successful Take-Ons", "Touches Attacking Penalty Area", 
    "Progressive Passes Received", "Tackles", "Interceptions", 
    "Blocks", "Clearances", "Aerials Won"
]
radar_columns_goalkeepers = [
    "PSxG-GA", "Goals Against", "Save Percentage", "Post-Shot Expected Goals per Shot on Target", 
    "Penalty Kicks Save Percentage","Launches Completion Percentage", "Clean Sheet Percentage", "Touches",  "Goal Kicks", 
    "Goal Kicks Average Length", "Crosses Stopped Percentage", "Defensive Action Outside Penalty Area", 
    "Average Distance of Defensive Action"
]

@st.cache_data
def get_similar_players_cosine(selected_player, df, n_top=10):
    df = df.set_index('Player Name')
    features = df.select_dtypes(include=[np.number]).drop(columns=['Age'])
    
    # Check if the selected player exists in the DataFrame
    if selected_player not in df.index:
        raise ValueError(f"{selected_player} not found in the DataFrame")

    # Get the feature data
    selected_player_features = features.loc[selected_player].values.reshape(1, -1)

    # Calculate the similarity using Cosine Similarity (cosine similarity ranges from -1 to 1)
    similarities = cosine_similarity(selected_player_features, features)[0]

    # Cosine similarity is normally between 0 and 1, but we want higher similarity to be better,
    # so we subtract from 1 to invert the scale (0 = worst similarity, 1 = best similarity)
    df['Similarity'] = similarities

    # Remove the selected player from the final list of similar players
    similar_players = df.drop(selected_player)

    # Sort and get the top `n_top` similar players
    similar_players = similar_players[['Similarity']].sort_values(by='Similarity', ascending=False).head(n_top)
    similar_players = similar_players.reset_index()
    similar_players['Rank'] = range(1, len(similar_players) + 1)

    # Merge with the relevant columns to get additional player information
    df = df.reset_index()
    relevant_columns = ['Player Name', 'Position', 'Team', 'Competition', 'Age', 'Nationality']
    similar_players = similar_players.merge(df[relevant_columns], on='Player Name')

    # Return the relevant columns
    similar_players = similar_players[['Rank', 'Player Name', 'Position', 'Team', 'Competition', 'Age', 'Nationality']]
    return similar_players


def consolidate_player_data(df):
    consolidated_df = df.groupby('Player Name').agg({
        'Position': 'first',
        'Team': lambda x: ', '.join(x.unique()),
        'Competition': lambda x: ', '.join(x.unique()),
        'Age': 'first',
        'Nationality': 'first',
        **{col: 'sum' for col in df.select_dtypes(include=[np.number]).columns if col != 'Age'}
    }).reset_index()
    return consolidated_df

@st.cache_data
def apply_filters(df, apply_age_filter, age_range, apply_nationality_filter, nationalities, apply_competition_filter, competition):
    # Apply filters to the dataset
    if apply_age_filter:
        df = df[(df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1])]

    if apply_nationality_filter and nationalities:
        df = df[df['Nationality'].isin(nationalities)]

    if apply_competition_filter and competition:
        # Split the competitions for each player and filter accordingly
        # Ensure the 'Competition' column is split into lists of competitions
        df['Competition'] = df['Competition'].apply(
            lambda comps: comps.split(', ') if isinstance(comps, str) else (comps if isinstance(comps, list) else [])
        )
        # Filter players based on whether they are involved in any of the selected competitions
        df = df[df['Competition'].apply(lambda comps: any(league in comps for league in competition))]

    return df

def calculate_per90_stats(df, radar_columns):
    df_per90 = df.copy()
    for column in radar_columns:
        if column not in ["Save Percentage", "Penalty Kicks Save Percentaage", "Clean Sheet Percentage", "Launches Completion Percentage", "Crosses Stopped Percentage"]:
            df_per90[column + '_per90'] = df.apply(
                lambda row: row[column] / row['90s'] if row['90s'] > 0 else 0, axis=1
            )
        else:
            df_per90[column + '_per90'] = df[column]
    return df_per90

def calculate_percentiles(df, radar_columns):
    df_percentiles = pd.DataFrame()
    df_percentiles['Player Name'] = df['Player Name']
    df_percentiles['Position'] = df['Position']
    df_percentiles['Team'] = df['Team']
    for column in radar_columns:
        if df[column].nunique() > 1:
            df_percentiles[column] = df[column].apply(lambda x: percentileofscore(df[column], x))
        else:
            df_percentiles[column] = 50
    return df_percentiles

def create_radar_plot(df_percentiles, player_names, radar_columns):
    radar_columns_per90 = [col + '_per90' if col not in ["Save Percentage", "Penalty Kicks Save Percentaage", "Clean Sheet Percentage", "Launches Completion Percentage", "Crosses Stopped Percentage"] else col for col in radar_columns]
    scatter_objects = []
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'pink', 'brown', 'cyan', 'magenta', 'yellow']
    
    for idx, player_name in enumerate(player_names):
        player_data = df_percentiles[df_percentiles['Player Name'] == player_name].iloc[0]
        player_values = player_data[radar_columns_per90].values
        
        player_scatter = go.Scatterpolar(
            r=player_values,
            theta=radar_columns,
            fill='toself',
            name=f"{player_name} ({player_data['Position']}, {player_data['Team']})",
            line=dict(color=colors[idx % len(colors)])  # Use distinct colors
        )
        
        scatter_objects.append(player_scatter)
    
    fig = go.Figure()
    fig.add_traces(scatter_objects)
    
    fig.update_layout(
        title_text='Radar Chart Comparison (Percentile Rankings per 90 Minutes)',
        height=500,
        width=800,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[0, 20, 40, 60, 80],
                ticktext=['0%', '20%', '40%', '60%', '80%'],
                tickfont=dict(color='black')  # Change the color of the radial axis labels
            ),
            angularaxis=dict(
                visible=True
            )
        ),
        showlegend=True
    )
    st.plotly_chart(fig)

def main():
    st.title("Football Player Analysis Tool 24-25")

    st.markdown("""
     ### Football Player Analysis 24-25
        
        This app focuses on identifying **players with similar playing styles and tactical roles**, rather than comparing raw output 
        
        Similarity is based on **per-90 normalized, role-defining metrics** (e.g. ball progression, defensive involvement, chance creation)
        
        You can:
        - find **stylistically similar players**,
        - compare up to **5 players head-to-head**,
        - explore player profiles using **radar charts based on percentile rankings**.
        
        ⚠️ *Radar charts appear further down the page after generating — please scroll to view them.*
        
        **Limitations**
        - Top 5 European leagues only  
        - Minimum 450 minutes played  
        - Primary focus on **style and role**, not performance prediction
        """)

    # Recommendation section for Outfield Players or Goalkeepers
    recommendation_type = st.selectbox(
        "Choose the type of recommendation:", 
        ["Outfield Players", "Goalkeepers"]
    )

    if recommendation_type in ["Outfield Players", "Goalkeepers"]:
        # Choose the appropriate dataframe based on the selection
        if recommendation_type == "Outfield Players":
            df = combined_df
            consolidated_df = consolidate_player_data(combined_df)
        else:  # Goalkeepers
            df = goalkeeper_combined_df
            consolidated_df = consolidate_player_data(goalkeeper_combined_df)

        # Prepare player information for dropdown
        player_info = [
            f"{row['Player Name']} ({row['Team']}, {row['Position']})" 
            for idx, row in consolidated_df.iterrows()
        ]
        player_name_map = dict(zip(player_info, consolidated_df['Player Name'].values))

        selected_players_info = st.selectbox(
            "Select a player to find similar players:", 
            player_info
        )
        selected_player = player_name_map[selected_players_info]
        
        # Filters
        n_top = st.number_input(
            "Number of similar players to display:", 
            min_value=1, 
            max_value=50, 
            value=10
        )
        apply_age_filter = st.checkbox("Filter by Age Range")
        apply_nationality_filter = st.checkbox("Filter by Nationality")
        apply_competition_filter = st.checkbox("Filter by Competition")

        # Initialize default values for filters
        age_range = (16, 50)  # Default range if no filter is applied
        nationalities = []
        competition = []

        # Get filter values
        if apply_age_filter:
            age_range = st.slider(
                "Select age range of similar players:", 
                min_value=16, 
                max_value=50, 
                value=(18, 30)
            )

        if apply_nationality_filter:
            nationalities = st.multiselect(
                "Select nationality of similar players:", 
                options=consolidated_df['Nationality'].unique()
            )

        if apply_competition_filter:
            # Extract unique individual leagues for the dropdown
            all_competitions = set()
            for comps in consolidated_df['Competition'].dropna():
                all_competitions.update(comps.split(', '))  # Split multi-league values into separate leagues

            competition = st.multiselect(
                "Select Competition of similar players:", 
                sorted(all_competitions)  # Show sorted unique leagues
            )

        # Apply filters and calculate similar players
        if st.button("Find Similar Players"):
            try:
                filtered_df = apply_filters(consolidated_df, apply_age_filter, age_range, apply_nationality_filter, nationalities, apply_competition_filter, competition)
                
                if selected_player not in filtered_df['Player Name'].values:
                    selected_player_data = consolidated_df[consolidated_df['Player Name'] == selected_player]
                    filtered_df = pd.concat([filtered_df, selected_player_data], ignore_index=True)
                
                similar_players = get_similar_players_cosine(selected_player, filtered_df, n_top=n_top)
                
                if similar_players.empty:
                    st.warning("No players found with the specified criteria.")
                else:
                    st.write(f"Similar players to {selected_player}:")
                    st.dataframe(similar_players, hide_index=True)
            except ValueError as e:
                st.error(str(e))

    # Radar chart section
    st.sidebar.title("Player Radar Chart Comparison")

    data_type = st.sidebar.radio("Select player type:", ["Outfield Players", "Goalkeepers"])

    # Cached function to load data
    # Caching only once for efficient use
    @st.cache_data
    def load_data(filepath):
        return pd.read_parquet(filepath, engine="pyarrow")

    # Load DataFrame for selected type
    if data_type == "Outfield Players":
        radar_columns = radar_columns_outfield
        df = load_data(outfield_file_path)  # <-- optimized, cached data loading
    elif data_type == "Goalkeepers":
        radar_columns = radar_columns_goalkeepers
        df = load_data(goalkeeper_file_path)  # <-- optimized, cached data loading
        if '90s' not in df.columns:
            st.error("The DataFrame does not contain a '90s' column.")
            return
        
    df_per90 = calculate_per90_stats(df, radar_columns)
    radar_columns_per90 = [col + '_per90' if col not in ["Save Percentage", "Penalty Kicks Save Percentaage", "Clean Sheet Percentage", "Launches Completion Percentage", "Crosses Stopped Percentage"] else col for col in radar_columns]
    df_percentiles = calculate_percentiles(df_per90, radar_columns_per90)
    
    player_info = [
        f"{row['Player Name']} ({row['Team']}, {row['Position']})"
        for idx, row in df.iterrows()
    ]
    
    player_name_map = dict(zip(player_info, df['Player Name'].values))
    
    selected_players_info = st.sidebar.multiselect(
        f"Select {data_type.lower()} to compare:", 
        player_info, 
        default=[player_info[0]]
    )
    
    selected_players = [player_name_map[info] for info in selected_players_info]
    
    if st.sidebar.button('Generate Radar Plot'):
        if selected_players:
            create_radar_plot(df_percentiles, selected_players, radar_columns)

if __name__ == "__main__":
    main()

# File paths
data_dir = "data/"
outfield_df = pd.read_parquet(data_dir + "outfield_df.parquet")
goalkeeper_df = pd.read_parquet(data_dir + "goalkeepers_df.parquet")

# Aggregate player data
@st.cache_data
def aggregate_player_data(df):
    return df.groupby("Player Name").agg({
        "Position": "first",
        "Team": lambda x: ", ".join(x.unique()),
        "Competition": lambda x: ", ".join(x.unique()),
        "Age": "first",  # Age is already provided
        "Nationality": "first",
        **{col: "sum" for col in df.select_dtypes(include=[np.number]).columns if col != "Age"}
    }).reset_index()

outfield_df, goalkeeper_df = map(aggregate_player_data, [outfield_df, goalkeeper_df])

# Add Player Info column
for df in [outfield_df, goalkeeper_df]:
    df["Player Info"] = df.apply(lambda row: f"{row['Player Name']} ({row['Team']}, {row['Position']})", axis=1)

# List of negative stats (higher value is worse)
negative_stats = [
    "Yellow Cards", "Red Cards", "Errors", "Miscontrols", "Dispossessed", "Second Yellow Cards", 
    "Fouls Commited", "Offsides", "Penalty Kicks Conceded", "Own Goals", "Aerials Lost", 
    "Goals Against", "Goals Against/90", "Penalty Kicks Allowed", "Penalty Kicks Missed", 
    "Free Kick Goals Against", "Corner Kick Goals Against", "Own Goals Scored Against Goalkeeper", 
    "Shots on Target Against"
]

# Streamlit UI
st.title("Head-to-Head Player Comparison")

# Select player type
player_type = st.radio("Select Player Type", ["Outfield Players", "Goalkeepers"])
df_used = outfield_df if player_type == "Outfield Players" else goalkeeper_df

# Player selection
player_options = df_used["Player Info"].tolist()
selected_players_info = []

# Always show selection for 2 players
selected_players_info.append(st.selectbox("Select Player 1", player_options, key="player_1"))
selected_players_info.append(st.selectbox("Select Player 2", player_options, key="player_2"))

# Expandable section for more players
with st.expander("➕ Add More Players"):
    for i in range(3, 6):
        player = st.selectbox(f"Select Player {i}", ["None"] + player_options, key=f"player_{i}")
        if player != "None":
            selected_players_info.append(player)

# Map selected info to player names
player_name_map = dict(zip(df_used["Player Info"], df_used["Player Name"]))
selected_players = [player_name_map[info] for info in selected_players_info]

# Display player details side by side
if selected_players:
    st.markdown("### Player Information")
    cols = st.columns(len(selected_players))
    details_df = df_used[df_used["Player Name"].isin(selected_players)]
    
    for col, player in zip(cols, selected_players):
        details = details_df[details_df["Player Name"] == player].iloc[0]
        with col:
            st.markdown(f"**{player}** ({details['Team']}, {details['Position']})")
            st.text(f"Age: {details['Age']}")
            st.text(f"Nationality: {details['Nationality']}")
            st.text(f"Competition: {details['Competition']}")

# Choose stats to compare
excluded_columns = ['Player Name', 'Player Info', 'Team', 'Position', 'Nationality', 'Competition', 'Year Born', 'Age']
stat_columns = [col for col in df_used.columns if col not in excluded_columns]
selected_stats = st.multiselect("Select Stats to Compare", stat_columns)

# Choose between raw stats or per 90 minutes
stat_type = st.radio("Select Data Type", ["Raw Stats", "Per 90 Minutes"])

# Convert to per 90 stats if selected
if stat_type == "Per 90 Minutes":
    for col in selected_stats:
        # Skip columns that are already per 90 minutes stats (contains ' per 90s' or '/90')
        if " per 90s" not in col and "/90" not in col:
            # Skip percentage columns (assuming percentage columns have "Percentage" in their names)
            if "Percentage" not in col:
                # Divide stats by the specific player's 90s value (row-wise)
                df_used[col] = df_used.apply(lambda row: row[col] / row["90s"] if row["90s"] != 0 else np.nan, axis=1)

# Filter selected players and create comparison table
@st.cache_data
def get_filtered_player_data(df, selected_players, selected_stats):
    """Retrieve and format the comparison table for selected players."""
    player_data = df[df["Player Name"].isin(selected_players)]
    comparison_df = player_data.set_index("Player Name")[selected_stats].T

    def format_number(val):
        """Format numbers: floats with 1 decimal, per 90s with 2 decimals."""
        return f"{val:.2f}" if isinstance(val, float) and ("90" in selected_stats or "Per 90" in selected_stats) else f"{val:.1f}" if isinstance(val, float) else val

    return comparison_df.applymap(format_number)

# Use the optimized function
comparison_df = get_filtered_player_data(df_used, selected_players, selected_stats)

# Format table column names
comparison_df.columns = [f"{player} ({df_used[df_used['Player Name'] == player]['Team'].values[0]}, {df_used[df_used['Player Name'] == player]['Position'].values[0]})"
                         for player in comparison_df.columns]

# Highlight best values (positive stats - green, negative stats - red, no color for equal values)
def highlight_best(s):
    is_negative = s.name in negative_stats
    max_val = s.max() if not is_negative else s.min()
    
    # Only highlight when the values are different
    return [
        'background-color: lightgreen' if v == max_val and s.nunique() > 1 else
        'background-color: lightcoral' if (is_negative and v == max_val) and s.nunique() > 1 else
        ''  # No color for equal values
        for v in s
    ]

# Apply the styling row-wise (axis=1) since stats are rows
styled_df = comparison_df.style.apply(highlight_best, axis=1)

# Display comparison table
if not selected_players:
    st.warning("⚠️ Please select at least one player to compare.")
else:
    st.write("### Player Comparison")
    st.dataframe(styled_df)
