# ⚽ Football Player Analysis Tool

## 📚 Table of Contents
- [TL;DR](#tldr)
- [🔗 Streamlit App](#-streamlit-app)
- [🎯 Project Overview](#-project-overview)
- [🔄 Data Collection & Preprocessing](#-data-collection--preprocessing)
- [💡 App Features & Functionality](#-app-features--functionality)
- [⚠️ Limitations & Future Improvements](#️-limitations--future-improvements)
- [📢 Disclaimer](#-disclaimer)

---

## TL;DR

- Interactive football analytics app built with **Streamlit**
- Focus on **player style and tactical role**, not raw statistical output
- Uses **per-90 normalised, role-defining metrics**
- Player similarity identified via **unsupervised learning (PCA + cosine similarity)**
- Designed for **player type comparison and replacement analysis**
- Includes:
  - Similar Player Finder
  - Percentile-based Radar Charts
  - Custom Head-to-Head Comparisons
  - Leaderboard of players with the highest values of selected metrics
- Data sourced from **FBref (Top 5 European leagues)**

---

## 🔗 Streamlit App
👉 https://scouting25.streamlit.app/

---

## 🎯 Project Overview

This project focuses on analysing football players based on their **playing style and tactical role**, rather than raw statistical output alone.

The core idea is **player type comparison and replacement**: identifying players who behave similarly on the pitch, even if their statistical production differs due to team context, tactical system or role within the squad. This makes the tool particularly useful when evaluating potential replacements across different teams and leagues.

Traditional player comparisons often rely on metrics such as goals, assists or defensive totals. However, these numbers are highly **context-dependent** and strongly influenced by factors like team strength, the teams tactical instructions and even the minutes played. As a result, raw output can be misleading when comparing players from different environments.

To address this, all comparisons in this app are based on **per-90 normalised, role-defining metrics** such as ball progression, defensive involvement and chance creation. This shifts the focus from *how much* a player produces to *how* a player contributes within a team structure.

Unsupervised machine learning is used to uncover statistical similarities without relying on predefined roles or positions. The goal is not to predict performance, but to identify **tactical profiles and playing tendencies**, allowing for more nuanced player evaluation.

The similarity model is designed as a starting point rather than a final judgement. It generates a shortlist of stylistically comparable players, with the option to display a larger number of similar profiles if desired. These players can then be explored in more detail using the app’s additional features. Radar profiles provide further insight into role similarity, while head-to-head comparisons and leaderboards allow users to assess differences in actual output, efficiency and statistical production.
---

## 🔄 Data Collection & Preprocessing

Player statistics were scraped from the  
[FBref Big 5 European Leagues Stats page](https://fbref.com/en/comps/Big5/Big-5-European-Leagues-Stats).

After extracting the individual stat tables, the data was cleaned and processed while retaining the original raw values.

### Dataset Structure
- Data is limited to **league matches** from the **Top 5 European leagues**
- Players are split into two groups:
  - **Outfield players**
  - **Goalkeepers**

Playing-time and team-level statistics were excluded where they did not meaningfully represent individual player behaviour.

### Preprocessing for Similarity Modelling
To reduce noise and contextual bias:
- Players were required to have played **at least the equivalent of 5 full matches**
- All count-based statistics were **normalised per 90 minutes** to ensure comparability across different playing times
- Redundant or **outcome-heavy** metrics were removed
- **RobustScaler** was used to normalise features while limiting the impact of outliers
- **PCA** was applied to reduce dimensionality while retaining ~90% of explained variance

Player similarity is calculated using **cosine similarity**, which compares the *direction* of statistical profiles rather than absolute magnitude. This approach proved more effective than distance-based methods (e.g. euclidian distance) for capturing stylistic similarity.

---

## 💡 App Features & Functionality

### 🔎 Similar Player Finder
- Select player type (Outfield / Goalkeeper)
- Filter by age, nationality, competition, position group
- Identify stylistically similar players based on PCA-transformed, style-defining metrics
  
### 📊 Radar Chart Comparison
- Percentile-based radar charts calculated by each individual role 
- Supports multiple players simultaneously

### 🆚 Head-to-Head Comparison
- Compare different players across selected metrics
- Toggle between raw stats and per-90 values
- Automatically highlights the better value (reversed for metrics where lower is better)
  
### 🏆 Stat Leaderboard
- Rank players by any metric (raw or per-90)
- Filters for age, nationality, competition, position group
- Optional minutes (90s) filter to exclude small-sample players and ensure reliable comparisons
---

## ⚠️ Limitations & Future Improvements

### Current Limitations
- Only includes league data from the **Top 5 European leagues**
- Minimum playing time threshold applies
- Role definitions may still show positional bias

### Planned Improvements
- Further role specification (e.g. CB,LB instead of DF)
- Having an additional output related model after the initial role similarity
- Inclusion of additional leagues
- Optional market value integration

---

## 📢 Disclaimer

This project is intended **solely for educational and personal portfolio use**.

It is **not affiliated with or endorsed by FBref.com, Sports Reference LLC, or their data providers**.

All data used in this project was sourced from publicly available pages on FBref.com and remains the intellectual property of its respective owners.
