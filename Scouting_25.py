import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import percentileofscore

# File paths
data_dir = "data"
goalkeeper_file_path = data_dir + "goalkeeper_combined_df.csv"
combined_file_path = data_dir + "combined_df.csv"
outfield_file_path = data_dir + "outfield_df.csv"
goalkeeper_file_path = data_dir + "goalkeeper_df.csv"

# Load datasets
goalkeeper_combined_df = pd.read_csv(goalkeeper_file_path)
combined_df = pd.read_csv(combined_file_path)

# Radar chart columns for outfield players
radar_columns_outfield = [
    "G-PK", "npxG", "Shots", "Assists", "xAG", "npxG+xAG", "SCA", 
    "Passes_Attempted", "Pass_Completion_%", "Progressive_Passes", 
    "Progressive_Carries", 'Take_Ons_Successful', 'Touches_Attacking_Penalty_Area', 
    'Progressive_Passes_Received', "Tackles", "Interceptions", 
    "Blocks", "Clearances", "Aerial_Duels_Won"
]

# Radar chart columns for goalkeepers
radar_columns_goalkeepers = [
    "Post-Shot_xG_+/-", "Goals_Against", "Save_%", "Post-Shot_xG_per_Shot_on_Target", 
    "Penalty_Kicks_Save_%", "Clean_Sheet_%", "Touches", "Launch_%", "Goal_Kicks", 
    "Average_Length_Goal_Kicks", "Crosses_Stopped_%", "Actions_Outside_Penalty_Area", 
    "Defensive_Actions_Average_Distance"
]

# Functions for player recommendations
def get_similar_players_euclidean(selected_player, df, n_top=10):
    df = df.set_index('Player')
    features = df.select_dtypes(include=[np.number]).drop(columns=['Age'])
    
    if selected_player not in df.index:
        raise ValueError(f"{selected_player} not found in the DataFrame")

    selected_player_features = features.loc[selected_player].values.reshape(1, -1)
    distances = euclidean_distances(selected_player_features, features)
    df['Similarity'] = -distances.flatten()
    similar_players = df.drop(selected_player)
    similar_players = similar_players[['Similarity']].sort_values(by='Similarity', ascending=False).head(n_top)
    similar_players = similar_players.reset_index()
    similar_players['Rank'] = range(1, len(similar_players) + 1)
    df = df.reset_index()
    relevant_columns = ['Player', 'Position', 'Team', 'Competition', 'Age', 'Nationality']
    similar_players = similar_players.merge(df[relevant_columns], on='Player')
    similar_players = similar_players[['Rank', 'Player', 'Position', 'Team', 'Competition', 'Age', 'Nationality']]
    return similar_players

def consolidate_player_data(df):
    consolidated_df = df.groupby('Player').agg({
        'Position': 'first',
        'Team': lambda x: ', '.join(x.unique()),
        'Competition': lambda x: ', '.join(x.unique()),
        'Age': 'first',
        'Nationality': 'first',
        **{col: 'sum' for col in df.select_dtypes(include=[np.number]).columns if col != 'Age'}
    }).reset_index()
    return consolidated_df

def calculate_per90_stats(df, radar_columns):
    df_per90 = df.copy()
    for column in radar_columns:
        if column not in ["Save_%", "Penalty_Kicks_Save_%", "Clean_Sheet_%", "Launch_%", "Crosses_Stopped_%"]:
            df_per90[column + '_per90'] = df.apply(
                lambda row: row[column] / row['90s'] if row['90s'] > 0 else 0, axis=1
            )
        else:
            df_per90[column + '_per90'] = df[column]
    return df_per90

def calculate_percentiles(df, radar_columns):
    df_percentiles = pd.DataFrame()
    df_percentiles['Player'] = df['Player']
    df_percentiles['Position'] = df['Position']
    df_percentiles['Team'] = df['Team']
    for column in radar_columns:
        if df[column].nunique() > 1:
            df_percentiles[column] = df[column].apply(lambda x: percentileofscore(df[column], x))
        else:
            df_percentiles[column] = 50
    return df_percentiles

def create_radar_plot(df_percentiles, player_names, radar_columns):
    radar_columns_per90 = [col + '_per90' if col not in ["Save_%", "Penalty_Kicks_Save_%", "Clean_Sheet_%", "Launch_%", "Crosses_Stopped_%"] else col for col in radar_columns]
    scatter_objects = []
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'pink', 'brown', 'cyan', 'magenta', 'yellow']
    
    for idx, player_name in enumerate(player_names):
        player_data = df_percentiles[df_percentiles['Player'] == player_name].iloc[0]
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

# Streamlit app
def main():
    st.title("Football Recommendation System")

    # Recommendation section
    recommendation_type = st.selectbox(
        "Choose the type of recommendation:", 
        ["General Players", "Goalkeepers"]
    )

    if recommendation_type == "General Players":
        # Prepare player information for dropdown
        combined_df_consolidated = consolidate_player_data(combined_df)
        player_info = [
            f"{row['Player']} ({row['Team']}, {row['Position']})" 
            for idx, row in combined_df_consolidated.iterrows()
        ]
        player_name_map = dict(zip(player_info, combined_df_consolidated['Player'].values))
        
        selected_players_info = st.selectbox(
            "Select a player to find similar players:", 
            player_info
        )
        
        selected_player = player_name_map[selected_players_info]
        n_top = st.number_input(
            "Number of similar players to display:", 
            min_value=1, 
            max_value=50, 
            value=10
        )
        
        apply_age_filter = st.checkbox("Filter by Age Range")
        apply_nationality_filter = st.checkbox("Filter by Nationality")
        apply_team_filter = st.checkbox("Filter by Team")

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
                options=combined_df_consolidated['Nationality'].unique()
            )

        if apply_team_filter:
            teams = st.multiselect(
                "Select team of similar players:", 
                options=combined_df_consolidated['Team'].unique()
            )

        if st.button("Find Similar Players"):
            try:
                similar_players = get_similar_players_euclidean(selected_player, combined_df_consolidated, n_top=n_top)
                
                if apply_age_filter:
                    similar_players = similar_players[
                        (similar_players['Age'] >= age_range[0]) & 
                        (similar_players['Age'] <= age_range[1])
                    ]
                
                if apply_nationality_filter:
                    if nationalities:
                        similar_players = similar_players[similar_players['Nationality'].isin(nationalities)]
                
                if apply_team_filter:
                    if teams:
                        similar_players = similar_players[similar_players['Team'].isin(teams)]
                
                if similar_players.empty:
                    st.warning("No players found with the specified criteria.")
                else:
                    st.write(f"Similar players to {selected_player}:")
                    st.dataframe(similar_players, hide_index=True)
            except ValueError as e:
                st.error(str(e))

    elif recommendation_type == "Goalkeepers":
        # Prepare goalkeeper information for dropdown    
        goalkeeper_info = [
            f"{row['Player']} ({row['Team']}, {row['Position']})"
            for idx, row in goalkeeper_combined_df.iterrows()
        ]
        goalkeeper_name_map = dict(zip(goalkeeper_info, goalkeeper_combined_df['Player'].values))
        
        selected_goalkeeper_info = st.selectbox(
            "Select a goalkeeper to find similar goalkeepers:",
            goalkeeper_info
        )
        
        selected_goalkeeper = goalkeeper_name_map[selected_goalkeeper_info]
        n_top = st.number_input(
            "Number of similar goalkeepers to display:",
            min_value=1,
            max_value=50,
            value=10
        )

        apply_age_filter = st.checkbox("Filter by Age Range")
        apply_nationality_filter = st.checkbox("Filter by Nationality")
        apply_team_filter = st.checkbox("Filter by Team")

        if apply_age_filter:
            age_range = st.slider(
                "Select age range of similar goalkeepers:",
                min_value=16,
                max_value=50,
                value=(18, 30)
            )

        if apply_nationality_filter:
            nationalities = st.multiselect(
                "Select nationality of similar goalkeepers:",
                options=goalkeeper_combined_df['Nationality'].unique()
            )

        if apply_team_filter:
            teams = st.multiselect(
                "Select team of similar goalkeepers:",
                options=goalkeeper_combined_df['Team'].unique()
            )

        if st.button("Find Similar Goalkeepers"):
            try:
                similar_goalkeepers = get_similar_players_euclidean(selected_goalkeeper, goalkeeper_combined_df, n_top=n_top)
                
                if apply_age_filter:
                    similar_goalkeepers = similar_goalkeepers[
                        (similar_goalkeepers['Age'] >= age_range[0]) & 
                        (similar_goalkeepers['Age'] <= age_range[1])
                    ]
                
                if apply_nationality_filter:
                    if nationalities:
                        similar_goalkeepers = similar_goalkeepers[similar_goalkeepers['Nationality'].isin(nationalities)]
                
                if apply_team_filter:
                    if teams:
                        similar_goalkeepers = similar_goalkeepers[similar_goalkeepers['Team'].isin(teams)]
                
                if similar_goalkeepers.empty:
                    st.warning("No goalkeepers found with the specified criteria.")
                else:
                    st.write(f"Similar goalkeepers to {selected_goalkeeper}:")
                    st.dataframe(similar_goalkeepers, hide_index=True)
            except ValueError as e:
                st.error(str(e))

    # Radar chart section
    st.sidebar.title("Player Radar Chart Comparison")

    data_type = st.sidebar.radio("Select player type:", ["Outfield Players", "Goalkeepers"])

    if data_type == "Outfield Players":
        radar_columns = radar_columns_outfield
        df = pd.read_csv(outfield_file_path)
    elif data_type == "Goalkeepers":
        radar_columns = radar_columns_goalkeepers
        df = pd.read_csv(goalkeeper_file_path)

    if '90s' not in df.columns:
        st.error("The DataFrame does not contain a '90s' column.")
        return
    
    df_per90 = calculate_per90_stats(df, radar_columns)
    radar_columns_per90 = [col + '_per90' if col not in ["Save_%", "Penalty_Kicks_Save_%", "Clean_Sheet_%", "Launch_%", "Crosses_Stopped_%"] else col for col in radar_columns]
    df_percentiles = calculate_percentiles(df_per90, radar_columns_per90)
    
    player_info = [
        f"{row['Player']} ({row['Team']}, {row['Position']})"
        for idx, row in df.iterrows()
    ]
    
    player_name_map = dict(zip(player_info, df['Player'].values))
    
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
