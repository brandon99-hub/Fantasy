import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

# Initialize session state components
if 'db' not in st.session_state:
    from database import FPLDatabase
    st.session_state.db = FPLDatabase()

if 'minutes_model' not in st.session_state:
    from models.minutes_model import MinutesPredictor
    st.session_state.minutes_model = MinutesPredictor()

if 'points_model' not in st.session_state:
    from models.points_model import PointsPredictor
    st.session_state.points_model = PointsPredictor()

def main():
    st.title("🔍 Player Analysis")
    st.markdown("Analyze player predictions, form, and performance trends")
    
    try:
        # Get players data
        players_df = st.session_state.db.get_players_with_stats()
        
        if players_df.empty:
            st.error("No player data available. Please update data first.")
            return
        
        # Sidebar filters
        st.sidebar.header("Filters")
        
        # Position filter
        positions = ['All'] + list(players_df['position'].unique())
        selected_position = st.sidebar.selectbox("Position", positions)
        
        # Team filter
        teams = ['All'] + list(players_df['team_name'].unique())
        selected_team = st.sidebar.selectbox("Team", teams)
        
        # Price range filter
        min_price = float(players_df['now_cost'].min() / 10)
        max_price = float(players_df['now_cost'].max() / 10)
        price_range = st.sidebar.slider(
            "Price Range (£m)", 
            min_value=min_price, 
            max_value=max_price, 
            value=(min_price, max_price),
            step=0.1
        )
        
        # Apply filters
        filtered_df = players_df.copy()
        
        if selected_position != 'All':
            filtered_df = filtered_df[filtered_df['position'] == selected_position]
        
        if selected_team != 'All':
            filtered_df = filtered_df[filtered_df['team_name'] == selected_team]
        
        filtered_df = filtered_df[
            (filtered_df['now_cost'] >= price_range[0] * 10) &
            (filtered_df['now_cost'] <= price_range[1] * 10)
        ]
        
        if filtered_df.empty:
            st.warning("No players match the selected filters.")
            return
        
        # Main content tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Predictions", "📊 Player Comparison", "📈 Form Analysis", "🔍 Player Details"])
        
        with tab1:
            show_predictions_tab(filtered_df)
        
        with tab2:
            show_comparison_tab(filtered_df)
        
        with tab3:
            show_form_analysis_tab(filtered_df)
        
        with tab4:
            show_player_details_tab(filtered_df)
    
    except Exception as e:
        st.error(f"Error loading player analysis: {str(e)}")

def show_predictions_tab(players_df):
    """Show AI predictions for players"""
    st.subheader("🎯 AI Predictions")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        # Generate predictions button
        if st.button("🧠 Generate Predictions", type="primary"):
            with st.spinner("Generating predictions..."):
                try:
                    # Get minutes predictions
                    minutes_pred = st.session_state.minutes_model.predict_minutes(players_df)
                    
                    # Get points predictions
                    points_pred = st.session_state.points_model.predict_points(players_df)
                    
                    if not points_pred.empty:
                        st.session_state['predictions'] = points_pred
                        st.success("Predictions generated!")
                    else:
                        st.error("Failed to generate predictions")
                
                except Exception as e:
                    st.error(f"Error generating predictions: {str(e)}")
    
    with col1:
        # Display predictions if available
        if 'predictions' in st.session_state and not st.session_state['predictions'].empty:
            predictions = st.session_state['predictions']
            
            # Top predictions
            st.markdown("#### Top Predicted Performers")
            
            # Sort options
            sort_options = {
                "Predicted Points": "predicted_points",
                "Points per Million": "points_per_million",
                "Expected Minutes": "expected_minutes"
            }
            
            sort_by = st.selectbox("Sort by:", list(sort_options.keys()))
            sort_col = sort_options[sort_by]
            
            top_predictions = predictions.nlargest(20, sort_col)
            
            # Display table with formatting
            display_cols = [
                'web_name', 'position', 'team_name', 'now_cost', 
                'predicted_points', 'expected_minutes', 'start_probability',
                'points_per_million', 'risk_category'
            ]
            
            # Format the display
            formatted_predictions = top_predictions.copy()
            formatted_predictions['now_cost'] = formatted_predictions['now_cost'] / 10
            formatted_predictions['predicted_points'] = formatted_predictions['predicted_points'].round(1)
            formatted_predictions['expected_minutes'] = formatted_predictions['expected_minutes'].round(0)
            formatted_predictions['start_probability'] = (formatted_predictions['start_probability'] * 100).round(0)
            formatted_predictions['points_per_million'] = formatted_predictions['points_per_million'].round(2)
            
            # Rename columns for display
            column_renames = {
                'web_name': 'Player',
                'position': 'Pos',
                'team_name': 'Team',
                'now_cost': 'Price (£m)',
                'predicted_points': 'Pred Points',
                'expected_minutes': 'Exp Minutes',
                'start_probability': 'Start %',
                'points_per_million': 'Pts/£m',
                'risk_category': 'Risk'
            }
            
            formatted_predictions = formatted_predictions[display_cols].rename(columns=column_renames)
            
            # Style the dataframe
            styled_df = formatted_predictions.style.format({
                'Price (£m)': '{:.1f}',
                'Pred Points': '{:.1f}',
                'Exp Minutes': '{:.0f}',
                'Start %': '{:.0f}%',
                'Pts/£m': '{:.2f}'
            })
            
            st.dataframe(styled_df, use_container_width=True, height=600)
            
            # Visualizations
            col1_viz, col2_viz = st.columns(2)
            
            with col1_viz:
                st.markdown("#### Predicted Points vs Price")
                fig = px.scatter(
                    predictions, 
                    x='now_cost', 
                    y='predicted_points',
                    color='position',
                    size='expected_minutes',
                    hover_data=['web_name', 'team_name'],
                    title="Value vs Predicted Performance"
                )
                fig.update_xaxes(title="Price (£0.1m)")
                fig.update_yaxes(title="Predicted Points")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2_viz:
                st.markdown("#### Start Probability Distribution")
                fig = px.histogram(
                    predictions,
                    x='start_probability',
                    color='position',
                    nbins=20,
                    title="Player Start Probability by Position"
                )
                fig.update_xaxes(title="Start Probability")
                fig.update_yaxes(title="Number of Players")
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("Click 'Generate Predictions' to see AI-powered player predictions.")

def show_comparison_tab(players_df):
    """Show player comparison tools"""
    st.subheader("📊 Player Comparison")
    
    # Player selection
    st.markdown("#### Select Players to Compare")
    
    # Multi-select for players
    player_options = players_df['web_name'].tolist()
    selected_players = st.multiselect(
        "Choose players:", 
        player_options,
        default=player_options[:3] if len(player_options) >= 3 else player_options
    )
    
    if not selected_players:
        st.warning("Please select at least one player to compare.")
        return
    
    # Get selected player data
    comparison_df = players_df[players_df['web_name'].isin(selected_players)].copy()
    
    # Comparison metrics
    st.markdown("#### Comparison Table")
    
    comparison_cols = [
        'web_name', 'position', 'team_name', 'now_cost', 'total_points',
        'form', 'points_per_game', 'selected_by_percent', 'goals_scored',
        'assists', 'clean_sheets', 'bonus', 'ict_index'
    ]
    
    # Format for display
    display_comparison = comparison_df[comparison_cols].copy()
    display_comparison['now_cost'] = display_comparison['now_cost'] / 10
    display_comparison = display_comparison.round(2)
    
    # Rename columns
    comparison_renames = {
        'web_name': 'Player',
        'position': 'Position',
        'team_name': 'Team',
        'now_cost': 'Price (£m)',
        'total_points': 'Total Points',
        'form': 'Form',
        'points_per_game': 'PPG',
        'selected_by_percent': 'Ownership %',
        'goals_scored': 'Goals',
        'assists': 'Assists',
        'clean_sheets': 'CS',
        'bonus': 'Bonus',
        'ict_index': 'ICT'
    }
    
    display_comparison = display_comparison.rename(columns=comparison_renames)
    st.dataframe(display_comparison.set_index('Player'), use_container_width=True)
    
    # Comparison charts
    if len(selected_players) > 1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Performance Comparison")
            metrics_to_compare = ['total_points', 'form', 'points_per_game', 'ict_index']
            
            fig = go.Figure()
            for i, player in enumerate(selected_players):
                player_data = comparison_df[comparison_df['web_name'] == player].iloc[0]
                values = [player_data[metric] for metric in metrics_to_compare]
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=['Total Points', 'Form', 'PPG', 'ICT Index'],
                    fill='toself',
                    name=player
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True)
                ),
                showlegend=True,
                title="Multi-dimensional Performance Comparison"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Value Analysis")
            
            fig = px.bar(
                comparison_df,
                x='web_name',
                y=['total_points', 'form'],
                title="Points vs Form Comparison",
                barmode='group'
            )
            fig.update_xaxes(title="Player")
            fig.update_yaxes(title="Points")
            st.plotly_chart(fig, use_container_width=True)

def show_form_analysis_tab(players_df):
    """Show player form analysis"""
    st.subheader("📈 Form Analysis")
    
    # Get historical data for form analysis
    try:
        history_df = st.session_state.db.get_player_history_data(limit_gws=10)
        
        if history_df.empty:
            st.warning("No historical data available for form analysis.")
            return
        
        # Form trends
        st.markdown("#### Recent Form Trends")
        
        # Top form players
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Best Form Players (Last 5 GWs)**")
            top_form = players_df.nlargest(10, 'form')[['web_name', 'team_name', 'position', 'form', 'total_points']]
            st.dataframe(top_form, use_container_width=True)
        
        with col2:
            st.markdown("**Form Distribution by Position**")
            fig = px.box(
                players_df,
                x='position',
                y='form',
                title="Form Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Individual player form tracking
        st.markdown("#### Individual Player Form")
        
        # Select player for detailed form analysis
        selected_player_form = st.selectbox(
            "Select player for detailed form analysis:",
            players_df['web_name'].tolist()
        )
        
        if selected_player_form:
            player_id = players_df[players_df['web_name'] == selected_player_form]['id'].iloc[0]
            player_history = history_df[history_df['element'] == player_id].sort_values('round')
            
            if not player_history.empty:
                # Create form trend chart
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=player_history['round'],
                    y=player_history['total_points'],
                    mode='lines+markers',
                    name='Points per GW',
                    line=dict(color='blue', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=player_history['round'],
                    y=player_history['minutes'],
                    mode='lines+markers',
                    name='Minutes',
                    yaxis='y2',
                    line=dict(color='green', width=2)
                ))
                
                fig.update_layout(
                    title=f"{selected_player_form} - Form Trend",
                    xaxis_title="Gameweek",
                    yaxis_title="Points",
                    yaxis2=dict(
                        title="Minutes",
                        overlaying='y',
                        side='right'
                    ),
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Form statistics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_points = player_history['total_points'].mean()
                    st.metric("Avg Points/GW", f"{avg_points:.1f}")
                
                with col2:
                    avg_minutes = player_history['minutes'].mean()
                    st.metric("Avg Minutes", f"{avg_minutes:.0f}")
                
                with col3:
                    games_started = (player_history['minutes'] >= 60).sum()
                    total_games = len(player_history)
                    start_rate = (games_started / total_games * 100) if total_games > 0 else 0
                    st.metric("Start Rate", f"{start_rate:.0f}%")
                
                with col4:
                    consistency = player_history['total_points'].std()
                    st.metric("Consistency (σ)", f"{consistency:.1f}")
            
            else:
                st.info(f"No historical data available for {selected_player_form}")
    
    except Exception as e:
        st.error(f"Error in form analysis: {str(e)}")

def show_player_details_tab(players_df):
    """Show detailed player information"""
    st.subheader("🔍 Player Details")
    
    # Player selection
    selected_player = st.selectbox(
        "Select a player:",
        players_df['web_name'].tolist()
    )
    
    if not selected_player:
        return
    
    player_data = players_df[players_df['web_name'] == selected_player].iloc[0]
    
    # Player header
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(f"# {player_data['web_name']}")
        st.markdown(f"**{player_data['position']} - {player_data['team_name']}**")
    
    # Key stats
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Price", f"£{player_data['now_cost'] / 10:.1f}m")
    
    with col2:
        st.metric("Total Points", f"{player_data['total_points']}")
    
    with col3:
        st.metric("Form", f"{player_data['form']:.1f}")
    
    with col4:
        st.metric("Ownership", f"{player_data['selected_by_percent']:.1f}%")
    
    with col5:
        st.metric("PPG", f"{player_data['points_per_game']:.1f}")
    
    # Detailed stats
    st.markdown("#### Season Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Attacking Stats**")
        stats_data = {
            "Goals": player_data.get('goals_scored', 0),
            "Assists": player_data.get('assists', 0),
            "Bonus Points": player_data.get('bonus', 0),
            "ICT Index": player_data.get('ict_index', 0)
        }
        
        for stat, value in stats_data.items():
            st.write(f"• {stat}: {value}")
    
    with col2:
        st.markdown("**Defensive Stats**")
        def_stats = {
            "Clean Sheets": player_data.get('clean_sheets', 0),
            "Goals Conceded": player_data.get('goals_conceded', 0),
            "Saves": player_data.get('saves', 0),
            "Yellow Cards": player_data.get('yellow_cards', 0)
        }
        
        for stat, value in def_stats.items():
            st.write(f"• {stat}: {value}")
    
    # Advanced metrics
    st.markdown("#### Advanced Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        influence = player_data.get('influence', 0)
        st.metric("Influence", f"{influence:.1f}")
    
    with col2:
        creativity = player_data.get('creativity', 0)
        st.metric("Creativity", f"{creativity:.1f}")
    
    with col3:
        threat = player_data.get('threat', 0)
        st.metric("Threat", f"{threat:.1f}")
    
    # Transfer trends
    if 'transfers_in_event' in player_data and 'transfers_out_event' in player_data:
        st.markdown("#### Transfer Activity")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Transfers In", f"{player_data['transfers_in_event']}")
        
        with col2:
            st.metric("Transfers Out", f"{player_data['transfers_out_event']}")
        
        with col3:
            net_transfers = player_data['transfers_in_event'] - player_data['transfers_out_event']
            st.metric("Net Transfers", f"{net_transfers:+d}")
    
    # Player news
    if 'news' in player_data and pd.notna(player_data['news']) and player_data['news'].strip():
        st.markdown("#### Latest News")
        st.info(player_data['news'])
    
    # Availability
    if 'chance_of_playing_this_round' in player_data:
        availability = player_data['chance_of_playing_this_round']
        if pd.notna(availability) and availability < 100:
            st.markdown("#### Availability")
            if availability == 0:
                st.error("Player is unavailable")
            elif availability < 50:
                st.warning(f"Doubtful to play ({availability}%)")
            else:
                st.info(f"Chance of playing: {availability}%")

if __name__ == "__main__":
    main()
