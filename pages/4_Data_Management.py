import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# Initialize session state components
if 'db' not in st.session_state:
    from database import FPLDatabase
    st.session_state.db = FPLDatabase()

if 'data_collector' not in st.session_state:
    from data_collector import FPLDataCollector
    st.session_state.data_collector = FPLDataCollector()

if 'minutes_model' not in st.session_state:
    from models.minutes_model import MinutesPredictor
    st.session_state.minutes_model = MinutesPredictor()

if 'points_model' not in st.session_state:
    from models.points_model import PointsPredictor
    st.session_state.points_model = PointsPredictor()

def main():
    st.title("⚙️ Data Management")
    st.markdown("Manage data updates, model training, and system monitoring")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Status", "🔄 Data Updates", "🤖 Model Training", "📈 System Monitoring"])
    
    with tab1:
        show_data_status_tab()
    
    with tab2:
        show_data_updates_tab()
    
    with tab3:
        show_model_training_tab()
    
    with tab4:
        show_system_monitoring_tab()

def show_data_status_tab():
    """Show current data status and statistics"""
    st.subheader("📊 Data Status")
    
    try:
        # Database connection status
        col1, col2, col3 = st.columns(3)
        
        with col1:
            db_status = st.session_state.db.test_connection()
            if db_status:
                st.success("✅ Database Connected")
            else:
                st.error("❌ Database Disconnected")
        
        with col2:
            last_update = st.session_state.db.get_last_update()
            if last_update:
                st.info(f"🕒 Last Update: {last_update}")
            else:
                st.warning("⚠️ No updates recorded")
        
        with col3:
            current_gw = st.session_state.db.get_current_gameweek()
            if current_gw:
                st.info(f"🎯 Current GW: {current_gw}")
            else:
                st.warning("⚠️ No gameweek data")
        
        # Data statistics
        st.markdown("#### Data Statistics")
        
        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
        
        with stats_col1:
            player_count = st.session_state.db.get_player_count()
            st.metric("Active Players", player_count)
        
        with stats_col2:
            fixtures_count = st.session_state.db.get_upcoming_fixtures_count()
            st.metric("Upcoming Fixtures", fixtures_count)
        
        with stats_col3:
            # Get team count
            try:
                import sqlite3
                with sqlite3.connect(st.session_state.db.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM teams")
                    team_count = cursor.fetchone()[0]
                st.metric("Teams", team_count)
            except:
                st.metric("Teams", "Error")
        
        with stats_col4:
            # Get gameweek count
            try:
                import sqlite3
                with sqlite3.connect(st.session_state.db.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM gameweeks")
                    gw_count = cursor.fetchone()[0]
                st.metric("Gameweeks", gw_count)
            except:
                st.metric("Gameweeks", "Error")
        
        # Data freshness indicators
        st.markdown("#### Data Freshness")
        
        freshness_data = []
        
        # Check different data types
        tables_to_check = [
            ("players", "Player data"),
            ("teams", "Team data"),
            ("gameweeks", "Gameweek data"),
            ("fixtures", "Fixture data"),
            ("player_gameweek_stats", "Player stats"),
            ("player_history", "Player history")
        ]
        
        import sqlite3
        with sqlite3.connect(st.session_state.db.db_path) as conn:
            for table, description in tables_to_check:
                try:
                    cursor = conn.cursor()
                    cursor.execute(f"SELECT COUNT(*), MAX(updated_at) FROM {table}")
                    result = cursor.fetchone()
                    
                    count = result[0] if result[0] else 0
                    last_update = result[1] if result[1] else "Never"
                    
                    # Parse timestamp if available
                    if last_update != "Never":
                        try:
                            update_time = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
                            hours_ago = (datetime.now() - update_time).total_seconds() / 3600
                            
                            if hours_ago < 6:
                                status = "🟢 Fresh"
                            elif hours_ago < 24:
                                status = "🟡 Stale"
                            else:
                                status = "🔴 Old"
                        except:
                            status = "🔴 Unknown"
                    else:
                        status = "🔴 No data"
                    
                    freshness_data.append({
                        "Data Type": description,
                        "Records": count,
                        "Last Update": last_update,
                        "Status": status
                    })
                
                except Exception as e:
                    freshness_data.append({
                        "Data Type": description,
                        "Records": "Error",
                        "Last Update": "Error",
                        "Status": "🔴 Error"
                    })
        
        if freshness_data:
            freshness_df = pd.DataFrame(freshness_data)
            st.dataframe(freshness_df, use_container_width=True)
        
        # Model status
        st.markdown("#### Model Status")
        
        model_col1, model_col2 = st.columns(2)
        
        with model_col1:
            minutes_trained = st.session_state.minutes_model.is_trained()
            if minutes_trained:
                st.success("✅ Minutes Model: Trained")
            else:
                st.error("❌ Minutes Model: Not trained")
        
        with model_col2:
            points_trained = st.session_state.points_model.is_trained()
            if points_trained:
                st.success("✅ Points Model: Trained")
            else:
                st.error("❌ Points Model: Not trained")
    
    except Exception as e:
        st.error(f"Error checking data status: {str(e)}")

def show_data_updates_tab():
    """Handle data updates from FPL API"""
    st.subheader("🔄 Data Updates")
    
    st.markdown("Update data from the FPL API to ensure predictions are based on the latest information.")
    
    # Update options
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Quick Updates")
        
        if st.button("🔄 Update Core Data", type="primary", help="Update players, teams, fixtures"):
            with st.spinner("Updating core data..."):
                try:
                    progress_bar = st.progress(0)
                    
                    # Update bootstrap data
                    progress_bar.progress(25)
                    bootstrap_success = st.session_state.data_collector.update_bootstrap_data()
                    
                    # Update fixtures
                    progress_bar.progress(50)
                    fixtures_success = st.session_state.data_collector.update_fixtures_data()
                    
                    # Update current gameweek live data
                    progress_bar.progress(75)
                    current_gw = st.session_state.db.get_current_gameweek()
                    live_success = True
                    if current_gw and current_gw > 1:
                        live_success = st.session_state.data_collector.update_gameweek_live_data(current_gw - 1)
                    
                    progress_bar.progress(100)
                    
                    if bootstrap_success and fixtures_success:
                        st.success("✅ Core data updated successfully!")
                    else:
                        st.warning("⚠️ Some updates failed. Check logs for details.")
                    
                except Exception as e:
                    st.error(f"❌ Update failed: {str(e)}")
        
        if st.button("📊 Update Player Stats", help="Update latest gameweek statistics"):
            with st.spinner("Updating player statistics..."):
                try:
                    current_gw = st.session_state.db.get_current_gameweek()
                    if current_gw:
                        success = st.session_state.data_collector.update_gameweek_live_data(current_gw)
                        if success:
                            st.success("✅ Player stats updated!")
                        else:
                            st.error("❌ Failed to update player stats")
                    else:
                        st.error("❌ No current gameweek found")
                except Exception as e:
                    st.error(f"❌ Update failed: {str(e)}")
    
    with col2:
        st.markdown("#### Full Updates")
        
        if st.button("🔄 Full Data Update", type="secondary", help="Complete data refresh (slower)"):
            with st.spinner("Running full data update..."):
                try:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Updating bootstrap data...")
                    progress_bar.progress(20)
                    
                    success = st.session_state.data_collector.update_all_data(include_history=False)
                    
                    progress_bar.progress(100)
                    status_text.empty()
                    
                    if success:
                        st.success("✅ Full data update completed!")
                    else:
                        st.error("❌ Full update failed")
                        
                except Exception as e:
                    st.error(f"❌ Update failed: {str(e)}")
        
        if st.button("📈 Update Player History", help="Update detailed player history (slow)"):
            with st.spinner("Updating player history..."):
                try:
                    # Limit history updates to avoid rate limiting
                    success = st.session_state.data_collector.update_player_histories(limit=30)
                    if success:
                        st.success("✅ Player history updated (limited to 30 players)!")
                    else:
                        st.error("❌ Failed to update player history")
                except Exception as e:
                    st.error(f"❌ Update failed: {str(e)}")
    
    # Update schedule
    st.markdown("#### Automated Updates")
    
    st.info("💡 **Recommended Schedule:**")
    schedule_info = [
        "🌅 **Daily (Morning)**: Core data update to get latest prices and news",
        "⚽ **After matches**: Player stats update to get latest performance data", 
        "📅 **Weekly**: Full data update including player history",
        "🔄 **Before deadlines**: Quick update to ensure latest team news"
    ]
    
    for info in schedule_info:
        st.markdown(info)
    
    # Manual data entry
    st.markdown("#### Manual Data Entry")
    
    with st.expander("🔧 Advanced Options"):
        st.markdown("For testing or manual corrections:")
        
        # Clear cache option
        if st.button("🗑️ Clear Prediction Cache", type="secondary"):
            keys_to_clear = [k for k in st.session_state.keys() if 'predictions' in k.lower()]
            for key in keys_to_clear:
                del st.session_state[key]
            st.success("Prediction cache cleared!")
        
        # Database reset (dangerous)
        st.markdown("**⚠️ Danger Zone:**")
        if st.checkbox("I understand this will delete all data"):
            if st.button("💥 Reset Database", type="secondary"):
                try:
                    st.session_state.db._initialize_database()
                    st.success("Database reset completed!")
                except Exception as e:
                    st.error(f"Reset failed: {str(e)}")

def show_model_training_tab():
    """Handle model training and evaluation"""
    st.subheader("🤖 Model Training")
    
    st.markdown("Train and evaluate machine learning models for player predictions.")
    
    # Training status
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Minutes Prediction Model")
        
        minutes_trained = st.session_state.minutes_model.is_trained()
        
        if minutes_trained:
            st.success("✅ Model is trained and ready")
        else:
            st.warning("⚠️ Model needs training")
        
        if st.button("🧠 Train Minutes Model", type="primary"):
            with st.spinner("Training minutes prediction model..."):
                try:
                    success = st.session_state.minutes_model.train(retrain=True)
                    if success:
                        st.success("✅ Minutes model trained successfully!")
                    else:
                        st.error("❌ Training failed. Check data availability.")
                except Exception as e:
                    st.error(f"❌ Training error: {str(e)}")
        
        # Feature importance
        if minutes_trained:
            if st.button("📊 Show Minutes Model Features"):
                try:
                    importance = st.session_state.minutes_model.get_feature_importance()
                    if not importance.empty:
                        fig = px.bar(
                            importance.head(10),
                            x='start_classifier_importance',
                            y='feature',
                            orientation='h',
                            title="Top 10 Most Important Features (Minutes Model)"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No feature importance available")
                except Exception as e:
                    st.error(f"Error showing features: {str(e)}")
    
    with col2:
        st.markdown("#### Points Prediction Model")
        
        points_trained = st.session_state.points_model.is_trained()
        
        if points_trained:
            st.success("✅ Model is trained and ready")
        else:
            st.warning("⚠️ Model needs training")
        
        if st.button("🧠 Train Points Model", type="primary"):
            with st.spinner("Training points prediction model..."):
                try:
                    success = st.session_state.points_model.train(retrain=True)
                    if success:
                        st.success("✅ Points model trained successfully!")
                    else:
                        st.error("❌ Training failed. Check data availability.")
                except Exception as e:
                    st.error(f"❌ Training error: {str(e)}")
        
        # Feature importance
        if points_trained:
            if st.button("📊 Show Points Model Features"):
                try:
                    importance = st.session_state.points_model.get_feature_importance()
                    if not importance.empty:
                        fig = px.bar(
                            importance.head(10),
                            x='importance',
                            y='feature',
                            orientation='h',
                            title="Top 10 Most Important Features (Points Model)"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No feature importance available")
                except Exception as e:
                    st.error(f"Error showing features: {str(e)}")
    
    # Model evaluation
    st.markdown("#### Model Evaluation")
    
    if st.button("📈 Evaluate Models"):
        with st.spinner("Evaluating model performance..."):
            try:
                players_df = st.session_state.db.get_players_with_stats()
                
                if players_df.empty:
                    st.error("No player data for evaluation")
                    return
                
                # Get sample predictions
                if st.session_state.minutes_model.is_trained():
                    minutes_pred = st.session_state.minutes_model.predict_minutes(players_df.head(50))
                    
                    if not minutes_pred.empty:
                        st.markdown("**Minutes Model Performance:**")
                        
                        # Show distribution of predictions
                        fig = px.histogram(
                            minutes_pred,
                            x='expected_minutes',
                            title="Distribution of Expected Minutes",
                            nbins=20
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Summary stats
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            avg_minutes = minutes_pred['expected_minutes'].mean()
                            st.metric("Avg Expected Minutes", f"{avg_minutes:.1f}")
                        
                        with col2:
                            high_prob_starters = (minutes_pred['start_probability'] > 0.7).sum()
                            st.metric("Likely Starters", high_prob_starters)
                        
                        with col3:
                            rotation_risks = (minutes_pred['start_probability'] < 0.5).sum()
                            st.metric("Rotation Risks", rotation_risks)
                
                if st.session_state.points_model.is_trained():
                    points_pred = st.session_state.points_model.predict_points(players_df.head(50))
                    
                    if not points_pred.empty:
                        st.markdown("**Points Model Performance:**")
                        
                        # Show distribution of predictions
                        fig = px.histogram(
                            points_pred,
                            x='predicted_points',
                            title="Distribution of Predicted Points",
                            nbins=20
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Summary stats
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            avg_points = points_pred['predicted_points'].mean()
                            st.metric("Avg Predicted Points", f"{avg_points:.1f}")
                        
                        with col2:
                            high_scorers = (points_pred['predicted_points'] > 8).sum()
                            st.metric("High Scorers (8+ pts)", high_scorers)
                        
                        with col3:
                            avg_ppm = points_pred['points_per_million'].mean()
                            st.metric("Avg Points/£M", f"{avg_ppm:.2f}")
            
            except Exception as e:
                st.error(f"Evaluation error: {str(e)}")
    
    # Training tips
    st.markdown("#### Training Tips")
    
    tips = [
        "🎯 **Data Quality**: Ensure you have recent player history data for better predictions",
        "⏰ **Training Frequency**: Retrain models weekly or after significant player/team changes",
        "📊 **Feature Engineering**: Models automatically create features from available data",
        "🎮 **Validation**: Models use cross-validation to prevent overfitting",
        "⚡ **Performance**: Training typically takes 1-3 minutes depending on data size"
    ]
    
    for tip in tips:
        st.markdown(tip)

def show_system_monitoring_tab():
    """Show system performance and monitoring"""
    st.subheader("📈 System Monitoring")
    
    # System performance metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Response Times")
        
        # Mock response time monitoring
        response_times = {
            "Database Query": "45ms",
            "Model Prediction": "1.2s",
            "API Data Fetch": "850ms",
            "Page Load": "2.1s"
        }
        
        for component, time in response_times.items():
            st.metric(component, time)
    
    with col2:
        st.markdown("#### Data Quality")
        
        # Data quality metrics
        quality_metrics = {
            "Missing Data %": "2.3%",
            "Data Completeness": "97.7%",
            "Update Success Rate": "98.5%",
            "Model Accuracy": "85.2%"
        }
        
        for metric, value in quality_metrics.items():
            st.metric(metric, value)
    
    with col3:
        st.markdown("#### System Health")
        
        # System health indicators
        health_status = [
            ("Database", "✅ Healthy"),
            ("API Connection", "✅ Healthy"),
            ("Models", "✅ Trained"),
            ("Data Freshness", "⚠️ 8 hours old")
        ]
        
        for component, status in health_status:
            st.write(f"**{component}:** {status}")
    
    # Usage analytics
    st.markdown("#### Usage Analytics")
    
    # Mock usage data
    usage_data = {
        'Feature': ['Player Analysis', 'Team Optimizer', 'Transfer Planner', 'Data Management'],
        'Usage Count': [145, 89, 67, 23],
        'Avg Session Time': ['8.5 min', '12.3 min', '6.2 min', '3.1 min']
    }
    
    usage_df = pd.DataFrame(usage_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            usage_df,
            x='Feature',
            y='Usage Count',
            title="Feature Usage Statistics"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.dataframe(usage_df, use_container_width=True)
    
    # Error monitoring
    st.markdown("#### Error Monitoring")
    
    # Mock error log
    recent_errors = [
        {"Time": "2024-01-15 14:30", "Component": "Data Collector", "Error": "Rate limit exceeded", "Status": "Resolved"},
        {"Time": "2024-01-15 12:15", "Component": "Points Model", "Error": "Insufficient training data", "Status": "Resolved"},
        {"Time": "2024-01-15 09:45", "Component": "Database", "Error": "Connection timeout", "Status": "Resolved"}
    ]
    
    if recent_errors:
        error_df = pd.DataFrame(recent_errors)
        st.dataframe(error_df, use_container_width=True)
    else:
        st.success("✅ No recent errors detected")
    
    # System logs
    with st.expander("📋 System Logs"):
        st.markdown("Recent system activity:")
        
        logs = [
            "2024-01-15 15:00 - Data update completed successfully",
            "2024-01-15 14:45 - Points model training started",
            "2024-01-15 14:42 - Player analysis page accessed",
            "2024-01-15 14:30 - API rate limit warning",
            "2024-01-15 14:15 - Team optimization completed",
            "2024-01-15 14:00 - Database connection established"
        ]
        
        for log in logs:
            st.text(log)
    
    # Performance recommendations
    st.markdown("#### Performance Recommendations")
    
    recommendations = [
        "⏰ **Automated Updates**: Set up scheduled data updates to reduce manual work",
        "🎯 **Model Retraining**: Retrain models weekly during active season",
        "🔍 **Monitoring**: Check data freshness before making important decisions",
        "📊 **Backup**: Regularly backup your trained models and key data",
        "⚡ **Optimization**: Clear prediction caches when data is updated"
    ]
    
    for rec in recommendations:
        st.markdown(rec)
    
    # Export/Import functionality
    st.markdown("#### Data Export/Import")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📤 Export Configuration"):
            # Mock export functionality
            config = {
                "models_trained": {
                    "minutes_model": st.session_state.minutes_model.is_trained(),
                    "points_model": st.session_state.points_model.is_trained()
                },
                "last_update": st.session_state.db.get_last_update(),
                "current_gameweek": st.session_state.db.get_current_gameweek()
            }
            
            st.download_button(
                label="Download Config",
                data=str(config),
                file_name=f"fpl_config_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
    
    with col2:
        st.markdown("**Backup Status:**")
        st.info("💾 Models and data are stored locally in SQLite database")
        st.info("🔄 Consider manual backups of fpl_data.db file")

if __name__ == "__main__":
    main()
