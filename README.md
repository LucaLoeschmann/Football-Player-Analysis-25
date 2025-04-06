# ⚽ Football Player Analysis Tool

## Streamlit App  
👉 [Click here to try the app](https://scouting25.streamlit.app/)

## 🌟 Table of Contents
- [🌟 Table of Contents](#-table-of-contents)
- [🎯 Project Overview](#-project-overview)
- [🔄 Data Collection & Preprocessing](#-data-collection--preprocessing)
- [💡 App Features & Functionality](#-app-features--functionality)
- [⚠️ Limitations & Future Improvements](#️-limitations--future-improvements)
- [📢 Disclaimer](#-disclaimer)

---

## 🎯 Project Overview

As an avid football fan with a strong interest in statistics, I wanted to dig deeper and create a tool that would help me compare players in different ways — all in one place. While there are already several professional websites offering great insights, I often felt limited by the typical head-to-head comparisons of just a few stats between two players.

I wasn’t just interested in the raw numbers, but also in how data could enhance our understanding of player performance. I wanted to apply **unsupervised machine learning** to uncover statistical similarities between players — not based on gut feeling, but on actual performance metrics. I was especially curious to see whether these statistical similarities would match my own perceptions, or if the data would challenge what I thought I knew.

➡️ You can explore the app here: [https://scouting25.streamlit.app](https://scouting25.streamlit.app)

The general concept was to create:
- A **Similar Player Finder** with filters for age, nationality, and competition
- A **Head-to-Head Comparison** tool for custom stat-based matchups
- A **Radar Chart Visualizer** to intuitively compare multiple players across multiple dimensions

The dataset is focused on **league matches** from the **Top 5 European leagues** (Premier League, Serie A, La Liga, Bundesliga, Ligue 1).

---

## 🔄 Data Collection & Preprocessing

I scraped player statistics from the [FBref Big 5 European Leagues Stats page](https://fbref.com/en/comps/Big5/Big-5-European-Leagues-Stats). After extracting the individual stat tables, I cleaned and processed the data to make it usable while retaining the original raw values.

### Column Naming
FBref uses short codes for stats that are displayed in the tables, which I expanded where necessary to improve clarity.

### Dataset Organization
I merged the individual dataframes into one combined dataset and then split it into two separate dataframes:
- **Outfield players**
- **Goalkeepers**

These two categories were handled separately due to their vastly different statistical profiles. The final datasets contain:
- 138 performance metrics for outfield players  
- 62 performance metrics for goalkeepers

I excluded "Playing Time" stats as these are mostly team-based stats that do not necessarily reflect individual player quality.

### Similarity Model Preprocessing
For the similarity model, more preprocessing was required to reduce noise. Some stats were dropped if they were redundant, overly dependent on role-specific factors (e.g., being the team's penalty taker), or influenced by external decisions.

To reduce outliers, I:
- Set a threshold requiring each player to have played the equivalent of at least 3 full games
- Used `RobustScaler` to normalize data while minimizing the influence of extreme values
- Applied **PCA** to reduce dimensionality, retaining **90% of the explained variance**

For measuring similarity, I used **cosine similarity**, which compares the angle between vectors rather than absolute values. Although Euclidean distance might seem intuitive for comparing totals, cosine similarity proved more effective at capturing stylistic similarity. In the future, I might experiment with combining both methods.

---

## 💡 App Features & Functionality

The Streamlit app offers:

### 🔍 Similar Player Finder
- Choose player type (Outfield or Goalkeeper)
- Apply filters: Age, Nationality, Competition
- Get top N similar players based on PCA-transformed stats and cosine similarity

### 🆚 Head-to-Head Comparison
- Compare up to 5 players across selected metrics
- Choose between raw stats or per 90 minutes
- Highlights best values automatically 

### 📊 Radar Chart Comparison
- Visualize percentile ranks across multiple performance categories
- Supports up to 10 players
- Separate templates for outfield players and goalkeepers

---

## ⚠️ Limitations & Future Improvements

### Current Limitations
- Only includes league data from the **Top 5 European Leagues**
- Only players with **at least 270 minutes played** are considered
- Stats used for similarity are filtered to avoid noise but may still reflect positional bias
- Penalty-related stats are excluded or downweighted due to team-dependence

### Future Enhancements
- Explore a hybrid similarity metric (e.g., cosine + Euclidean)
- Add role-specific models (e.g., separate models for CBs, Wingers, etc.)
- Add more leagues
- Take the market value of the players into account 

---

## 📢 Disclaimer

This project is intended **solely for educational and personal portfolio use**.  
It is **not affiliated with or endorsed by FBref.com, Sports Reference LLC, or their data providers**.

Player data used in this app was sourced from publicly available web pages on FBref.com and is presented for analysis and research purposes.  
The app does **not reproduce a complete database**, does **not offer any form of commercial service**, and is **not intended to compete with FBref or any related platform**.

All data remains the intellectual property of its respective owners.
