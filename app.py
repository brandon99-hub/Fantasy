import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sqlite3
import numpy as np
from data_collector import FPLDataCollector
from database import FPLDatabase
from models.minutes_model import MinutesPredictor
from models.points_model import PointsPredictor

# Page config
st.set_page_config(
    page_title="FPL AI Optimizer",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'db' not in st.session_state:
    st.session_state.db = FPLDatabase()
    
if 'data_collector' not in st.session_state:
    st.session_state.data_collector = FPLDataCollector()
    
if 'minutes_model' not in st.session_state:
    st.session_state.minutes_model = MinutesPredictor()
    
if 'points_model' not in st.session_state:
    st.session_state.points_model = PointsPredictor()

def main():
    st.title("⚽ FPL AI Optimizer")
    st.markdown("### Fantasy Premier League optimization with ML-powered predictions")
    
    # Sidebar
    st.sidebar.title("Navigation")
    st.sidebar.markdown("Use the pages above to navigate through different features.")
    
    # Data status check
    with st.sidebar.expander("📊 Data Status", expanded=True):
        try:
            last_update = st.session_state.db.get_last_update()
            if last_update:
                st.success(f"Last updated: {last_update}")
            else:
                st.warning("No data available")
                
            # Quick data refresh
            if st.button("🔄 Refresh Data"):
                with st.spinner("Updating data..."):
                    success = st.session_state.data_collector.update_all_data()
                    if success:
                        st.success("Data updated successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to update data")
        except Exception as e:
            st.error(f"Database error: {str(e)}")
    
    # Main dashboard
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Players in Database",
            value=st.session_state.db.get_player_count(),
            delta="Active players"
        )
    
    with col2:
        current_gw = st.session_state.db.get_current_gameweek()
        st.metric(
            label="Current Gameweek", 
            value=current_gw if current_gw else "Unknown"
        )
    
    with col3:
        fixtures_count = st.session_state.db.get_upcoming_fixtures_count()
        st.metric(
            label="Upcoming Fixtures",
            value=fixtures_count
        )
    
    # Quick insights
    st.markdown("---")
    st.subheader("📈 Quick Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Top Form Players (Last 5 GWs)")
        try:
            top_form = st.session_state.db.get_top_form_players(5)
            if not top_form.empty:
                top_form_display = top_form[['web_name', 'team_name', 'position', 'form_points', 'total_points']]
                st.dataframe(top_form_display, use_container_width=True)
            else:
                st.info("No form data available")
        except Exception as e:
            st.error(f"Error loading form data: {str(e)}")
    
    with col2:
        st.markdown("#### Price Changes (Last 7 Days)")
        try:
            price_changes = st.session_state.db.get_recent_price_changes(7)
            if not price_changes.empty:
                # Color code rises and falls
                def color_price_change(val):
                    color = 'green' if val > 0 else 'red' if val < 0 else 'black'
                    return f'color: {color}'
                
                price_display = price_changes[['web_name', 'team_name', 'price_change', 'current_price']]
                styled_df = price_display.style.applymap(color_price_change, subset=['price_change'])
                st.dataframe(styled_df, use_container_width=True)
            else:
                st.info("No recent price changes")
        except Exception as e:
            st.error(f"Error loading price changes: {str(e)}")
    
    # Feature overview
    st.markdown("---")
    st.subheader("🎯 Platform Features")
    
    features_col1, features_col2 = st.columns(2)
    
    with features_col1:
        st.markdown("""
        #### 🤖 AI Predictions
        - **Minutes Probability Model**: Predicts likelihood of players starting/playing
        - **Points Prediction Model**: Expected points based on fixtures and form
        - **Risk Assessment**: Variance and ceiling analysis for each player
        """)
        
        st.markdown("""
        #### 📊 Player Analysis
        - Form analysis and trends
        - Fixture difficulty assessment
        - Comparison tools
        - Historical performance
        """)
    
    with features_col2:
        st.markdown("""
        #### ⚡ Team Optimization
        - Mathematical optimization using OR-Tools
        - Respects all FPL rules and constraints
        - Captain and formation optimization
        - Chip strategy recommendations
        """)
        
        st.markdown("""
        #### 🔄 Transfer Planning
        - Multi-week transfer planning
        - Hit calculation and optimization
        - Price change predictions
        - Wildcard timing suggestions
        """)
    
    # System status
    st.markdown("---")
    st.subheader("⚙️ System Status")
    
    status_col1, status_col2, status_col3 = st.columns(3)
    
    with status_col1:
        try:
            model_status = st.session_state.minutes_model.is_trained()
            if model_status:
                st.success("✅ Minutes Model: Ready")
            else:
                st.warning("⚠️ Minutes Model: Not trained")
        except:
            st.error("❌ Minutes Model: Error")
    
    with status_col2:
        try:
            model_status = st.session_state.points_model.is_trained()
            if model_status:
                st.success("✅ Points Model: Ready")
            else:
                st.warning("⚠️ Points Model: Not trained")
        except:
            st.error("❌ Points Model: Error")
    
    with status_col3:
        try:
            db_status = st.session_state.db.test_connection()
            if db_status:
                st.success("✅ Database: Connected")
            else:
                st.error("❌ Database: Disconnected")
        except:
            st.error("❌ Database: Error")

if __name__ == "__main__":
    main()
