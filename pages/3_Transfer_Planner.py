import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# Initialize session state components
if 'db' not in st.session_state:
    from database import FPLDatabase
    st.session_state.db = FPLDatabase()

if 'points_model' not in st.session_state:
    from models.points_model import PointsPredictor
    st.session_state.points_model = PointsPredictor()

if 'optimizer' not in st.session_state:
    from optimizer import FPLOptimizer
    st.session_state.optimizer = FPLOptimizer()

def main():
    st.title("🔄 Transfer Planner")
    st.markdown("Plan optimal transfers and manage your FPL squad across multiple gameweeks")
    
    try:
        # Get current players data
        players_df = st.session_state.db.get_players_with_stats()
        
        if players_df.empty:
            st.error("No player data available. Please update data first.")
            return
        
        # Sidebar settings
        st.sidebar.header("Transfer Settings")
        
        current_gw = st.session_state.db.get_current_gameweek()
        if not current_gw:
            current_gw = 1
        
        # Planning horizon
        planning_horizon = st.sidebar.slider("Planning Horizon (Gameweeks)", 1, 8, 3)
        
        # Transfer budget
        free_transfers = st.sidebar.number_input("Free Transfers Available", 0, 5, 1)
        max_hits = st.sidebar.number_input("Max Hits Allowed", 0, 10, 2)
        
        # Chip planning
        st.sidebar.subheader("Chip Planning")
        plan_wildcard = st.sidebar.checkbox("Plan Wildcard Usage")
        plan_free_hit = st.sidebar.checkbox("Plan Free Hit Usage")
        plan_bench_boost = st.sidebar.checkbox("Plan Bench Boost Usage")
        
        # Main content tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Transfer Recommendations", "📅 Multi-Week Planning", "💰 Hit Calculator", "🃏 Chip Strategy"])
        
        with tab1:
            show_transfer_recommendations_tab(players_df, free_transfers)
        
        with tab2:
            show_multi_week_planning_tab(players_df, planning_horizon, free_transfers, max_hits)
        
        with tab3:
            show_hit_calculator_tab(players_df)
        
        with tab4:
            show_chip_strategy_tab(players_df, current_gw)
    
    except Exception as e:
        st.error(f"Error in transfer planner: {str(e)}")

def show_transfer_recommendations_tab(players_df, free_transfers):
    """Show immediate transfer recommendations"""
    st.subheader("🎯 Transfer Recommendations")
    
    # Current team input
    st.markdown("#### Your Current Team")
    
    current_team_input = st.text_area(
        "Enter your current 15 players (one per line):",
        help="List your current squad players, one name per line",
        height=200
    )
    
    if not current_team_input:
        st.info("Please enter your current team to get transfer recommendations.")
        return
    
    # Parse current team
    current_player_names = [name.strip() for name in current_team_input.split('\n') if name.strip()]
    current_team_df = players_df[players_df['web_name'].isin(current_player_names)].copy()
    
    if len(current_team_df) != 15:
        st.warning(f"Found {len(current_team_df)} players. Please ensure you have exactly 15 players.")
        if len(current_team_df) > 0:
            st.write("Found players:")
            st.dataframe(current_team_df[['web_name', 'position', 'team_name', 'now_cost']])
        return
    
    st.success(f"Current team loaded: {len(current_team_df)} players")
    
    # Generate predictions if not available
    if st.button("🧠 Analyze Current Team", type="primary"):
        with st.spinner("Analyzing your team and generating recommendations..."):
            try:
                # Get predictions for all players
                predictions = st.session_state.points_model.predict_points(players_df)
                
                if predictions.empty:
                    st.error("Failed to generate predictions")
                    return
                
                st.session_state['transfer_predictions'] = predictions
                st.session_state['current_team_analysis'] = current_team_df
                
            except Exception as e:
                st.error(f"Error analyzing team: {str(e)}")
                return
    
    # Show recommendations if available
    if 'transfer_predictions' in st.session_state and 'current_team_analysis' in st.session_state:
        show_transfer_analysis(st.session_state['transfer_predictions'], 
                             st.session_state['current_team_analysis'], 
                             free_transfers)

def show_transfer_analysis(predictions, current_team_df, free_transfers):
    """Show detailed transfer analysis"""
    
    # Current team performance
    st.markdown("#### Current Team Analysis")
    
    current_predictions = predictions[predictions['web_name'].isin(current_team_df['web_name'])].copy()
    
    if current_predictions.empty:
        st.error("Could not analyze current team predictions")
        return
    
    # Team metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_predicted = current_predictions['predicted_points'].sum()
    total_cost = current_team_df['now_cost'].sum() / 10
    avg_ownership = current_predictions['selected_by_percent'].mean()
    
    with col1:
        st.metric("Predicted Points", f"{total_predicted:.1f}")
    
    with col2:
        st.metric("Team Value", f"£{total_cost:.1f}M")
    
    with col3:
        st.metric("Avg Ownership", f"{avg_ownership:.1f}%")
    
    with col4:
        budget_remaining = 100.0 - total_cost
        st.metric("Budget Remaining", f"£{budget_remaining:.1f}M")
    
    # Underperforming players
    st.markdown("#### Players to Consider Transferring Out")
    
    # Sort current team by predicted points (lowest first)
    underperformers = current_predictions.nsmallest(8, 'predicted_points')
    
    transfer_out_data = []
    for _, player in underperformers.iterrows():
        # Find better alternatives in same position and similar price
        position = player['position']
        price = player['now_cost']
        
        alternatives = predictions[
            (predictions['position'] == position) &
            (predictions['now_cost'] >= price - 5) &
            (predictions['now_cost'] <= price + 10) &
            (~predictions['web_name'].isin(current_team_df['web_name']))
        ].nlargest(3, 'predicted_points')
        
        if not alternatives.empty:
            best_alternative = alternatives.iloc[0]
            points_upgrade = best_alternative['predicted_points'] - player['predicted_points']
            
            if points_upgrade > 0.5:  # Only suggest if meaningful upgrade
                transfer_out_data.append({
                    'Player': player['web_name'],
                    'Position': position,
                    'Current Points': f"{player['predicted_points']:.1f}",
                    'Best Alternative': best_alternative['web_name'],
                    'Alternative Points': f"{best_alternative['predicted_points']:.1f}",
                    'Points Upgrade': f"+{points_upgrade:.1f}",
                    'Cost Change': f"£{(best_alternative['now_cost'] - player['now_cost'])/10:.1f}M"
                })
    
    if transfer_out_data:
        transfer_df = pd.DataFrame(transfer_out_data)
        st.dataframe(transfer_df, use_container_width=True)
    else:
        st.success("Your team looks well optimized! No obvious transfer targets.")
    
    # Hot picks
    st.markdown("#### Hot Transfer Targets")
    
    # Players not in current team, sorted by predicted points
    available_players = predictions[~predictions['web_name'].isin(current_team_df['web_name'])]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Best Value Picks**")
        value_picks = available_players.copy()
        value_picks['value_score'] = value_picks['predicted_points'] / (value_picks['now_cost'] / 10)
        top_value = value_picks.nlargest(5, 'value_score')
        
        for _, player in top_value.iterrows():
            ownership_indicator = "🔥" if player['selected_by_percent'] < 10 else ""
            st.write(f"• {ownership_indicator} **{player['web_name']}** ({player['position']}) - {player['predicted_points']:.1f} pts, £{player['now_cost']/10:.1f}M")
    
    with col2:
        st.markdown("**Form Players**")
        if 'form' in available_players.columns:
            form_picks = available_players.nlargest(5, 'form')
            
            for _, player in form_picks.iterrows():
                ownership_indicator = "🔥" if player['selected_by_percent'] < 10 else ""
                st.write(f"• {ownership_indicator} **{player['web_name']}** ({player['position']}) - {player['form']:.1f} form, {player['predicted_points']:.1f} pts")
    
    # Transfer suggestions based on free transfers
    st.markdown("#### Recommended Transfers")
    
    if free_transfers > 0:
        st.success(f"You have {free_transfers} free transfer{'s' if free_transfers > 1 else ''}!")
        
        if transfer_out_data:
            st.markdown("**Suggested moves:**")
            for i, transfer in enumerate(transfer_df.head(free_transfers).to_dict('records')):
                st.write(f"{i+1}. **{transfer['Player']}** → **{transfer['Best Alternative']}** ({transfer['Points Upgrade']} points)")
    else:
        st.warning("No free transfers available. Consider if any transfers are worth a -4 point hit.")

def show_multi_week_planning_tab(players_df, planning_horizon, free_transfers, max_hits):
    """Show multi-week transfer planning"""
    st.subheader("📅 Multi-Week Planning")
    
    st.info(f"Planning transfers for the next {planning_horizon} gameweeks")
    
    # Get fixture data for planning
    try:
        # This would need fixture data from the database
        # For now, show a simplified version
        
        st.markdown("#### Fixture Analysis")
        
        # Team fixture difficulty over next few weeks
        current_gw = st.session_state.db.get_current_gameweek() or 1
        
        # Mock fixture difficulty data (would come from database)
        st.markdown("**Upcoming Fixture Difficulty by Team:**")
        
        teams_df = players_df.groupby('team_name').agg({
            'web_name': 'count',
            'total_points': 'mean',
            'form': 'mean'
        }).rename(columns={'web_name': 'player_count', 'total_points': 'avg_points', 'form': 'avg_form'})
        
        teams_df['fixture_score'] = np.random.uniform(2, 5, len(teams_df))  # Mock data
        teams_df = teams_df.sort_values('fixture_score')
        
        # Display teams with best fixtures
        st.markdown("**Teams with Favorable Fixtures:**")
        good_fixtures = teams_df.head(8)
        
        for team, data in good_fixtures.iterrows():
            st.write(f"• **{team}** - Fixture difficulty: {data['fixture_score']:.1f}/5, Avg form: {data['avg_form']:.1f}")
        
        # Players from teams with good fixtures
        st.markdown("#### Players from Teams with Good Fixtures")
        
        good_fixture_teams = good_fixtures.index.tolist()
        good_fixture_players = players_df[players_df['team_name'].isin(good_fixture_teams)]
        
        # Top players from these teams
        top_fixture_players = good_fixture_players.nlargest(10, 'form')
        
        display_cols = ['web_name', 'position', 'team_name', 'form', 'total_points', 'now_cost', 'selected_by_percent']
        fixture_display = top_fixture_players[display_cols].copy()
        fixture_display['now_cost'] = fixture_display['now_cost'] / 10
        fixture_display.columns = ['Player', 'Pos', 'Team', 'Form', 'Points', 'Price £m', 'Own %']
        
        st.dataframe(fixture_display, use_container_width=True)
        
        # Transfer timeline planning
        st.markdown("#### Transfer Timeline")
        
        timeline_data = []
        for week in range(planning_horizon):
            gw = current_gw + week
            timeline_data.append({
                'Gameweek': f'GW{gw}',
                'Free Transfers': 1 if week > 0 else free_transfers,
                'Recommended Action': 'Monitor prices' if week == 0 else f'Plan transfer {week}',
                'Focus': 'Form players' if week < 2 else 'Fixture-based'
            })
        
        timeline_df = pd.DataFrame(timeline_data)
        st.dataframe(timeline_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error in multi-week planning: {str(e)}")

def show_hit_calculator_tab(players_df):
    """Calculate if transfers are worth taking hits"""
    st.subheader("💰 Hit Calculator")
    
    st.markdown("Calculate whether a transfer is worth taking a -4 point hit")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Transfer Out")
        
        # Player selection
        out_player = st.selectbox("Player to transfer out:", players_df['web_name'].tolist())
        
        if out_player:
            out_player_data = players_df[players_df['web_name'] == out_player].iloc[0]
            
            st.write(f"**Position:** {out_player_data['position']}")
            st.write(f"**Current price:** £{out_player_data['now_cost']/10:.1f}M")
            st.write(f"**Total points:** {out_player_data['total_points']}")
            st.write(f"**Form:** {out_player_data.get('form', 0):.1f}")
            
            # Expected points input
            out_expected = st.number_input("Expected points (next few GWs):", 0.0, 50.0, float(out_player_data.get('form', 5)), key="out_expected")
    
    with col2:
        st.markdown("#### Transfer In")
        
        in_player = st.selectbox("Player to transfer in:", players_df['web_name'].tolist())
        
        if in_player:
            in_player_data = players_df[players_df['web_name'] == in_player].iloc[0]
            
            st.write(f"**Position:** {in_player_data['position']}")
            st.write(f"**Current price:** £{in_player_data['now_cost']/10:.1f}M")
            st.write(f"**Total points:** {in_player_data['total_points']}")
            st.write(f"**Form:** {in_player_data.get('form', 0):.1f}")
            
            # Expected points input
            in_expected = st.number_input("Expected points (next few GWs):", 0.0, 50.0, float(in_player_data.get('form', 5)), key="in_expected")
    
    # Hit calculation
    if out_player and in_player and out_player != in_player:
        st.markdown("---")
        st.markdown("#### Hit Analysis")
        
        # Basic calculation
        points_difference = in_expected - out_expected
        hit_cost = 4  # Standard hit cost
        net_gain = points_difference - hit_cost
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Points Difference", f"+{points_difference:.1f}")
        
        with col2:
            st.metric("Hit Cost", f"-{hit_cost}")
        
        with col3:
            st.metric("Net Gain/Loss", f"{net_gain:+.1f}")
        
        with col4:
            breakeven = hit_cost / max(points_difference, 0.1)
            st.metric("Breakeven (GWs)", f"{breakeven:.1f}")
        
        # Recommendation
        if net_gain > 0:
            st.success(f"✅ **Recommended**: This transfer should gain you {net_gain:.1f} points")
        elif net_gain > -2:
            st.warning(f"⚠️ **Marginal**: This transfer loses {abs(net_gain):.1f} points but might be worth it")
        else:
            st.error(f"❌ **Not recommended**: This transfer would lose you {abs(net_gain):.1f} points")
        
        # Additional considerations
        st.markdown("#### Additional Considerations")
        
        considerations = [
            f"💰 **Budget**: Transfer frees up £{(out_player_data['now_cost'] - in_player_data['now_cost'])/10:.1f}M" if out_player_data['now_cost'] > in_player_data['now_cost'] else f"💰 **Budget**: Transfer costs £{(in_player_data['now_cost'] - out_player_data['now_cost'])/10:.1f}M",
            f"📈 **Ownership**: {in_player_data.get('selected_by_percent', 0):.1f}% vs {out_player_data.get('selected_by_percent', 0):.1f}%",
            f"🏆 **Position**: Both players play {in_player_data['position']}" if in_player_data['position'] == out_player_data['position'] else f"⚠️ **Position**: Position change from {out_player_data['position']} to {in_player_data['position']}"
        ]
        
        for consideration in considerations:
            st.markdown(consideration)

def show_chip_strategy_tab(players_df, current_gw):
    """Show chip usage strategy"""
    st.subheader("🃏 Chip Strategy")
    
    st.markdown("Plan your chip usage for maximum points gain")
    
    # Chip overview
    chips_info = {
        "Wildcard": {
            "description": "Make unlimited transfers for one gameweek",
            "uses": "2 per season (1st half + 2nd half)",
            "best_time": "International breaks, fixture swings, injury crises"
        },
        "Free Hit": {
            "description": "Make unlimited transfers for one gameweek, team reverts next week",
            "uses": "1 per season",
            "best_time": "Blank gameweeks, double gameweeks"
        },
        "Bench Boost": {
            "description": "Points from bench players count",
            "uses": "1 per season", 
            "best_time": "Double gameweeks when bench players likely to play"
        },
        "Triple Captain": {
            "description": "Captain scores triple points instead of double",
            "uses": "1 per season",
            "best_time": "Double gameweeks for reliable captains"
        }
    }
    
    # Display chip information
    for chip, info in chips_info.items():
        with st.expander(f"💡 {chip}"):
            st.write(f"**Description:** {info['description']}")
            st.write(f"**Uses:** {info['uses']}")
            st.write(f"**Best timing:** {info['best_time']}")
    
    # Chip timing recommendations
    st.markdown("#### Chip Timing Strategy")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Early Season Chips (GW1-19)**")
        early_recommendations = [
            "🃏 **First Wildcard**: Use around GW6-8 after initial template fails",
            "⭐ **Triple Captain**: Save for double gameweeks (typically GW26-29)",
            "🔄 **Free Hit**: Hold for blank gameweeks or major rotation",
            "📈 **Bench Boost**: Plan for double gameweeks with playing bench"
        ]
        
        for rec in early_recommendations:
            st.markdown(rec)
    
    with col2:
        st.markdown("**Late Season Chips (GW20-38)**")
        late_recommendations = [
            "🃏 **Second Wildcard**: Use for fixture swings or final push",
            "⭐ **Triple Captain**: Double gameweeks with premium players",
            "🔄 **Free Hit**: Blank gameweeks or when team structure broken",
            "📈 **Bench Boost**: Double gameweeks with strong bench"
        ]
        
        for rec in late_recommendations:
            st.markdown(rec)
    
    # Current chip recommendations
    st.markdown("#### Current Recommendations")
    
    # Mock recommendations based on gameweek
    if current_gw <= 10:
        st.info("🎯 **Early Season**: Hold chips for now. Monitor team performance and plan first wildcard around GW6-8.")
    elif current_gw <= 20:
        st.warning("⚠️ **Mid Season**: Consider wildcard if team structure is poor. Start planning for double gameweeks.")
    elif current_gw <= 30:
        st.success("✅ **Double Gameweek Season**: Perfect time for Triple Captain and Bench Boost if you have good players with 2 games.")
    else:
        st.error("🏁 **End Game**: Use remaining chips strategically. Free Hit can be valuable in final gameweeks.")
    
    # Chip value calculator
    st.markdown("#### Chip Value Calculator")
    
    selected_chip = st.selectbox("Select chip to analyze:", list(chips_info.keys()))
    
    if selected_chip == "Bench Boost":
        st.markdown("**Bench Boost Analysis**")
        st.write("Calculate expected points from your bench:")
        
        bench_players = []
        for i in range(4):
            player = st.selectbox(f"Bench player {i+1}:", ['None'] + players_df['web_name'].tolist(), key=f"bench_{i}")
            if player != 'None':
                bench_players.append(player)
        
        if bench_players:
            bench_data = players_df[players_df['web_name'].isin(bench_players)]
            expected_bench_points = bench_data['form'].sum() if 'form' in bench_data.columns else 0
            
            st.metric("Expected Bench Points", f"{expected_bench_points:.1f}")
            
            if expected_bench_points > 8:
                st.success("✅ Good time for Bench Boost!")
            elif expected_bench_points > 5:
                st.warning("⚠️ Moderate bench boost value")
            else:
                st.error("❌ Wait for better bench boost opportunity")
    
    elif selected_chip == "Triple Captain":
        st.markdown("**Triple Captain Analysis**")
        
        captain_candidate = st.selectbox("Captain candidate:", players_df['web_name'].tolist())
        
        if captain_candidate:
            captain_data = players_df[players_df['web_name'] == captain_candidate].iloc[0]
            expected_points = captain_data.get('form', 5)
            triple_points = expected_points * 3
            double_points = expected_points * 2
            bonus = triple_points - double_points
            
            st.metric("Expected Triple Captain Points", f"{triple_points:.1f}")
            st.metric("Bonus vs Normal Captain", f"+{bonus:.1f}")
            
            if bonus > 8:
                st.success("✅ Great Triple Captain opportunity!")
            elif bonus > 5:
                st.warning("⚠️ Decent Triple Captain value")
            else:
                st.error("❌ Wait for double gameweeks")

if __name__ == "__main__":
    main()
