import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from optimizer import FPLOptimizer
import json

# Initialize session state components
if 'db' not in st.session_state:
    from database import FPLDatabase
    st.session_state.db = FPLDatabase()

if 'points_model' not in st.session_state:
    from models.points_model import PointsPredictor
    st.session_state.points_model = PointsPredictor()

if 'optimizer' not in st.session_state:
    st.session_state.optimizer = FPLOptimizer()

def main():
    st.title("⚡ Team Optimizer")
    st.markdown("Build the optimal FPL team using mathematical optimization")
    
    try:
        # Get players data
        players_df = st.session_state.db.get_players_with_stats()
        
        if players_df.empty:
            st.error("No player data available. Please update data first.")
            return
        
        # Check if models are trained
        if not st.session_state.points_model.is_trained():
            st.warning("Points prediction model is not trained. Training now...")
            with st.spinner("Training prediction models..."):
                success = st.session_state.points_model.train()
                if not success:
                    st.error("Failed to train prediction models. Using basic optimization.")
        
        # Sidebar configuration
        st.sidebar.header("Optimization Settings")
        
        # Budget setting
        budget = st.sidebar.number_input(
            "Budget (£m)",
            min_value=80.0,
            max_value=120.0,
            value=100.0,
            step=0.1,
            help="Total budget for the squad"
        )
        
        # Formation selection
        formations = ["3-4-3", "3-5-2", "4-3-3", "4-4-2", "4-5-1", "5-3-2", "5-4-1"]
        selected_formation = st.sidebar.selectbox("Preferred Formation", formations)
        
        # Transfer settings
        st.sidebar.subheader("Transfer Settings")
        use_current_team = st.sidebar.checkbox("Optimize from current team", help="Use transfer optimization instead of full team selection")
        
        if use_current_team:
            free_transfers = st.sidebar.number_input("Free Transfers", min_value=0, max_value=5, value=1)
            use_wildcard = st.sidebar.checkbox("Use Wildcard", help="Ignore transfer limits")
        else:
            free_transfers = 1
            use_wildcard = False
        
        # Advanced options
        with st.sidebar.expander("Advanced Options"):
            min_predicted_minutes = st.slider("Min Expected Minutes", 0, 90, 30, help="Exclude players with very low expected minutes")
            exclude_injured = st.checkbox("Exclude Injured Players", value=True)
            max_ownership = st.slider("Max Ownership %", 0, 100, 100, help="Exclude highly owned players for differentials")
        
        # Main content
        tab1, tab2, tab3 = st.tabs(["🎯 Optimization", "📊 Team Analysis", "💡 Recommendations"])
        
        with tab1:
            show_optimization_tab(players_df, budget, selected_formation, use_current_team, 
                                free_transfers, use_wildcard, min_predicted_minutes, 
                                exclude_injured, max_ownership)
        
        with tab2:
            show_team_analysis_tab()
        
        with tab3:
            show_recommendations_tab(players_df)
    
    except Exception as e:
        st.error(f"Error in team optimizer: {str(e)}")

def show_optimization_tab(players_df, budget, formation, use_current_team, 
                         free_transfers, use_wildcard, min_predicted_minutes, 
                         exclude_injured, max_ownership):
    """Main optimization interface"""
    
    st.subheader("🎯 Team Optimization")
    
    # Generate predictions first
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if st.button("🧠 Generate Predictions", type="secondary"):
            with st.spinner("Generating player predictions..."):
                try:
                    predictions = st.session_state.points_model.predict_points(players_df)
                    if not predictions.empty:
                        st.session_state['optimization_predictions'] = predictions
                        st.success("Predictions generated!")
                    else:
                        st.error("Failed to generate predictions")
                except Exception as e:
                    st.error(f"Error generating predictions: {str(e)}")
    
    with col1:
        # Check if predictions are available
        if 'optimization_predictions' not in st.session_state:
            st.info("Generate predictions first to enable optimization.")
            return
        
        predictions = st.session_state['optimization_predictions'].copy()
        
        # Apply filters
        if exclude_injured:
            predictions = predictions[
                (predictions.get('status', 'a') == 'a') & 
                (predictions.get('chance_of_playing_this_round', 100) > 50)
            ]
        
        if min_predicted_minutes > 0:
            predictions = predictions[predictions['expected_minutes'] >= min_predicted_minutes]
        
        if max_ownership < 100:
            predictions = predictions[predictions.get('selected_by_percent', 0) <= max_ownership]
        
        st.info(f"Optimizing from {len(predictions)} eligible players")
    
    # Current team input (if using transfer optimization)
    current_team_ids = []
    if use_current_team:
        st.markdown("#### Current Team")
        
        # Simple text input for current team (could be enhanced with player selection)
        current_team_text = st.text_area(
            "Enter current team player names (one per line):",
            help="Enter the names of your current 15 players, one per line"
        )
        
        if current_team_text:
            current_team_names = [name.strip() for name in current_team_text.split('\n') if name.strip()]
            current_team_data = players_df[players_df['web_name'].isin(current_team_names)]
            current_team_ids = current_team_data['id'].tolist()
            
            if len(current_team_ids) != 15:
                st.warning(f"Found {len(current_team_ids)} players. Need exactly 15 for optimization.")
            else:
                st.success(f"Current team loaded: {len(current_team_ids)} players")
    
    # Optimize button
    if st.button("⚡ Optimize Team", type="primary"):
        if 'optimization_predictions' not in st.session_state:
            st.error("Please generate predictions first")
            return
        
        with st.spinner("Running optimization..."):
            try:
                # Prepare optimization parameters
                optimization_params = {
                    'players_df': predictions,
                    'budget': budget,
                    'formation': formation,
                    'use_wildcard': use_wildcard
                }
                
                if use_current_team and len(current_team_ids) == 15:
                    optimization_params['current_team'] = current_team_ids
                    optimization_params['free_transfers'] = free_transfers
                
                # Run optimization
                solution = st.session_state.optimizer.optimize_team(**optimization_params)
                
                if solution:
                    st.session_state['optimized_team'] = solution
                    st.success("Team optimization completed!")
                else:
                    st.error("Optimization failed. No feasible solution found.")
                    return
                
            except Exception as e:
                st.error(f"Optimization error: {str(e)}")
                return
    
    # Display optimized team
    if 'optimized_team' in st.session_state:
        solution = st.session_state['optimized_team']
        display_optimized_team(solution)

def display_optimized_team(solution):
    """Display the optimized team solution"""
    st.markdown("---")
    st.subheader("🏆 Optimized Team")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Cost", f"£{solution['total_cost']:.1f}M")
    
    with col2:
        st.metric("Budget Remaining", f"£{100.0 - solution['total_cost']:.1f}M")
    
    with col3:
        st.metric("Predicted Points", f"{solution['predicted_points']:.1f}")
    
    with col4:
        transfer_cost = solution.get('transfers', {}).get('cost', 0)
        st.metric("Transfer Cost", f"-{transfer_cost} pts" if transfer_cost > 0 else "Free")
    
    # Starting XI
    st.markdown("#### Starting XI")
    if solution['starting_xi']:
        starting_df = pd.DataFrame(solution['starting_xi'])
        
        # Add captain indicator
        captain_id = solution.get('captain', {}).get('id')
        starting_df['captain'] = starting_df['id'] == captain_id
        starting_df['display_name'] = starting_df.apply(
            lambda x: f"⭐ {x['name']}" if x['captain'] else x['name'], axis=1
        )
        
        # Format for display
        display_starting = starting_df[['display_name', 'position', 'team', 'cost', 'predicted_points']].copy()
        display_starting.columns = ['Player', 'Position', 'Team', 'Cost (£m)', 'Predicted Points']
        
        st.dataframe(display_starting, use_container_width=True)
    
    # Bench
    st.markdown("#### Bench")
    if solution['bench']:
        bench_df = pd.DataFrame(solution['bench'])
        display_bench = bench_df[['name', 'position', 'team', 'cost', 'predicted_points']].copy()
        display_bench.columns = ['Player', 'Position', 'Team', 'Cost (£m)', 'Predicted Points']
        
        st.dataframe(display_bench, use_container_width=True)
    
    # Transfers (if any)
    if solution.get('transfers', {}).get('in'):
        st.markdown("#### Transfers")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Transfers In:**")
            transfers_in = pd.DataFrame(solution['transfers']['in'])
            st.dataframe(transfers_in[['name', 'cost']], use_container_width=True)
        
        with col2:
            st.markdown("**Transfers Out:**")
            transfers_out = pd.DataFrame(solution['transfers']['out'])
            st.dataframe(transfers_out[['name', 'cost']], use_container_width=True)
    
    # Team visualization
    st.markdown("#### Formation Visualization")
    create_formation_viz(solution)

def create_formation_viz(solution):
    """Create a formation visualization"""
    try:
        # Group players by position
        starting_xi = solution.get('starting_xi', [])
        
        position_groups = {
            'GKP': [],
            'DEF': [],
            'MID': [],
            'FWD': []
        }
        
        for player in starting_xi:
            pos = player.get('position', 'MID')
            position_groups[pos].append(player)
        
        # Create formation layout
        fig = go.Figure()
        
        # Position coordinates (normalized 0-1)
        y_positions = {'GKP': 0.1, 'DEF': 0.3, 'MID': 0.6, 'FWD': 0.9}
        
        captain_id = solution.get('captain', {}).get('id')
        
        for pos, players in position_groups.items():
            if not players:
                continue
            
            y = y_positions[pos]
            num_players = len(players)
            
            # Distribute players evenly across width
            if num_players == 1:
                x_positions = [0.5]
            else:
                x_positions = [i / (num_players - 1) for i in range(num_players)]
            
            for i, player in enumerate(players):
                is_captain = player.get('id') == captain_id
                
                fig.add_trace(go.Scatter(
                    x=[x_positions[i]],
                    y=[y],
                    mode='markers+text',
                    marker=dict(
                        size=30 if is_captain else 20,
                        color='gold' if is_captain else 'lightblue',
                        line=dict(width=2, color='black')
                    ),
                    text=player.get('name', ''),
                    textposition='bottom center',
                    showlegend=False,
                    hovertemplate=f"<b>{player.get('name')}</b><br>"
                                f"Position: {player.get('position')}<br>"
                                f"Team: {player.get('team')}<br>"
                                f"Cost: £{player.get('cost', 0):.1f}m<br>"
                                f"Predicted: {player.get('predicted_points', 0):.1f} pts<br>"
                                f"{'(Captain)' if is_captain else ''}<extra></extra>"
                ))
        
        fig.update_layout(
            title="Team Formation",
            xaxis=dict(range=[0, 1], showticklabels=False, showgrid=False),
            yaxis=dict(range=[0, 1], showticklabels=False, showgrid=False),
            plot_bgcolor='green',
            width=600,
            height=400,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating formation visualization: {str(e)}")

def show_team_analysis_tab():
    """Show analysis of the optimized team"""
    st.subheader("📊 Team Analysis")
    
    if 'optimized_team' not in st.session_state:
        st.info("Optimize a team first to see analysis.")
        return
    
    solution = st.session_state['optimized_team']
    
    # Analyze the team
    analysis = st.session_state.optimizer.analyze_team_value(solution)
    
    if not analysis:
        st.error("Failed to analyze team.")
        return
    
    # Formation analysis
    st.markdown("#### Formation Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        formation_data = analysis.get('formation_analysis', {})
        st.metric("Formation", formation_data.get('formation', 'Unknown'))
        
        counts = formation_data.get('counts', {})
        for pos, count in counts.items():
            if pos != 'GKP':  # Skip GKP as it's always 1
                st.write(f"• {pos}: {count}")
    
    with col2:
        # Position distribution chart
        if 'position_breakdown' in analysis:
            pos_data = analysis['position_breakdown']
            positions = list(pos_data.keys())
            costs = [pos_data[pos]['total_cost'] for pos in positions]
            
            fig = px.pie(
                values=costs,
                names=positions,
                title="Budget Distribution by Position"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Value analysis
    st.markdown("#### Value Analysis")
    
    if 'value_analysis' in analysis:
        value_data = analysis['value_analysis']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Points per £M", f"{value_data.get('points_per_million', 0):.2f}")
        
        with col2:
            most_expensive = value_data.get('most_expensive', {})
            st.metric("Most Expensive", f"£{most_expensive.get('cost', 0):.1f}M")
            if most_expensive.get('name'):
                st.write(f"*{most_expensive['name']}*")
        
        with col3:
            cheapest = value_data.get('cheapest', {})
            st.metric("Cheapest", f"£{cheapest.get('cost', 0):.1f}M")
            if cheapest.get('name'):
                st.write(f"*{cheapest['name']}*")
    
    # Position breakdown table
    st.markdown("#### Position Breakdown")
    
    if 'position_breakdown' in analysis:
        pos_breakdown = []
        for pos, data in analysis['position_breakdown'].items():
            pos_breakdown.append({
                'Position': pos,
                'Players': data['count'],
                'Total Cost': f"£{data['total_cost']:.1f}M",
                'Avg Cost': f"£{data['avg_cost']:.1f}M",
                'Total Points': f"{data['total_points']:.1f}"
            })
        
        breakdown_df = pd.DataFrame(pos_breakdown)
        st.dataframe(breakdown_df, use_container_width=True)

def show_recommendations_tab(players_df):
    """Show team building recommendations"""
    st.subheader("💡 Recommendations")
    
    # General recommendations
    st.markdown("#### Strategic Recommendations")
    
    recommendations = [
        "🎯 **Formation Strategy**: 3-4-3 and 4-3-3 are popular for balanced attacking returns",
        "💰 **Budget Allocation**: Spend 65-70% on your starting XI, save budget for bench coverage",
        "⭐ **Captain Choice**: Select players with high predicted points and low rotation risk",
        "🔄 **Rotation Risk**: Avoid players from teams with heavy fixture congestion",
        "📈 **Form vs Fixtures**: Balance current form with upcoming fixture difficulty",
        "🏆 **Differential Strategy**: Consider low-owned players for rank climbing"
    ]
    
    for rec in recommendations:
        st.markdown(rec)
    
    # Data-driven insights
    st.markdown("#### Current Market Insights")
    
    try:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Best Value Players by Position**")
            
            for position in ['GKP', 'DEF', 'MID', 'FWD']:
                pos_players = players_df[players_df['position'] == position]
                if not pos_players.empty:
                    pos_players['value'] = pos_players['total_points'] / (pos_players['now_cost'] / 10)
                    best_value = pos_players.nlargest(3, 'value')
                    
                    st.markdown(f"**{position}:**")
                    for _, player in best_value.iterrows():
                        st.write(f"• {player['web_name']} ({player['team_name']}) - {player['value']:.1f} pts/£m")
        
        with col2:
            st.markdown("**Form Players to Watch**")
            
            # Top form players across positions
            top_form = players_df.nlargest(8, 'form')
            
            for _, player in top_form.iterrows():
                ownership = player.get('selected_by_percent', 0)
                differential = "🔥" if ownership < 10 else ""
                st.write(f"• {differential} {player['web_name']} ({player['position']}) - {player['form']:.1f} form")
    
    except Exception as e:
        st.error(f"Error generating insights: {str(e)}")
    
    # Captain recommendations
    if 'optimized_team' in st.session_state:
        st.markdown("#### Captain Recommendations")
        
        starting_xi_df = pd.DataFrame(st.session_state['optimized_team']['starting_xi'])
        if not starting_xi_df.empty:
            captain_analysis = st.session_state.optimizer.optimize_captain_choice(starting_xi_df)
            
            if captain_analysis:
                recommended = captain_analysis.get('recommended_captain', {})
                
                st.success(f"Recommended Captain: **{recommended.get('name')}** ({recommended.get('predicted_points', 0):.1f} predicted points)")
                
                alternatives = captain_analysis.get('alternatives', [])
                if alternatives:
                    st.markdown("**Alternative Options:**")
                    for alt in alternatives:
                        st.write(f"• {alt.get('web_name')} - {alt.get('predicted_points', 0):.1f} pts")

if __name__ == "__main__":
    main()
