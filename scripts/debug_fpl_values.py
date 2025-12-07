"""
Debug script to check what values are causing integer overflow
"""
import sys
from pathlib import Path
import requests

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

def check_fpl_data():
    """Fetch FPL data and check for values that might overflow"""
    print("Fetching FPL bootstrap data...")
    
    response = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/")
    data = response.json()
    
    print("\nChecking player data for large values...")
    players = data['elements']
    
    # Check for values that might overflow INTEGER (max: 2,147,483,647)
    INT_MAX = 2147483647
    
    issues = []
    
    for player in players:
        player_id = player['id']
        player_name = player['web_name']
        
        # Check fields we changed to INTEGER
        fields_to_check = {
            'now_cost': player.get('now_cost'),
            'cost_change_start': player.get('cost_change_start'),
            'cost_change_event': player.get('cost_change_event'),
            'total_points': player.get('total_points'),
            'minutes': player.get('minutes'),
            'goals_scored': player.get('goals_scored'),
            'assists': player.get('assists'),
            'clean_sheets': player.get('clean_sheets'),
            'goals_conceded': player.get('goals_conceded'),
            'own_goals': player.get('own_goals'),
            'penalties_saved': player.get('penalties_saved'),
            'penalties_missed': player.get('penalties_missed'),
            'yellow_cards': player.get('yellow_cards'),
            'red_cards': player.get('red_cards'),
            'saves': player.get('saves'),
            'bonus': player.get('bonus'),
            'bps': player.get('bps'),
            'starts': player.get('starts'),
        }
        
        for field, value in fields_to_check.items():
            if value is not None and abs(value) > INT_MAX:
                issues.append(f"  ❌ Player {player_name} ({player_id}): {field} = {value:,} (exceeds INTEGER)")
    
    if issues:
        print(f"\nFound {len(issues)} INTEGER overflow issues:")
        for issue in issues[:10]:  # Show first 10
            print(issue)
    else:
        print("No INTEGER overflow issues found in player data")
    
    print("\nChecking gameweek data for large values...")
    gameweeks = data['events']
    
    gw_issues = []
    for gw in gameweeks:
        gw_id = gw['id']
        
        fields_to_check = {
            'average_entry_score': gw.get('average_entry_score'),
            'highest_score': gw.get('highest_score'),
            'most_selected': gw.get('most_selected'),
            'most_transferred_in': gw.get('most_transferred_in'),
            'most_captained': gw.get('most_captained'),
            'most_vice_captained': gw.get('most_vice_captained'),
        }
        
        for field, value in fields_to_check.items():
            if value is not None and abs(value) > INT_MAX:
                gw_issues.append(f"  ❌ GW {gw_id}: {field} = {value:,} (exceeds INTEGER)")
    
    if gw_issues:
        print(f"\nFound {len(gw_issues)} INTEGER overflow issues in gameweeks:")
        for issue in gw_issues[:10]:
            print(issue)
    else:
        print("No INTEGER overflow issues found in gameweek data")
    
    # Show sample values
    print("\nSample player values:")
    sample_player = players[0]
    print(f"  Player: {sample_player['web_name']}")
    print(f"  now_cost: {sample_player.get('now_cost'):,}")
    print(f"  total_points: {sample_player.get('total_points'):,}")
    print(f"  transfers_in: {sample_player.get('transfers_in'):,}")
    print(f"  transfers_out: {sample_player.get('transfers_out'):,}")
    
    print("\nSample gameweek values:")
    if gameweeks:
        sample_gw = gameweeks[0]
        print(f"  GW: {sample_gw['id']}")
        print(f"  average_entry_score: {sample_gw.get('average_entry_score')}")
        print(f"  highest_score: {sample_gw.get('highest_score')}")
        print(f"  most_selected: {sample_gw.get('most_selected'):,}")
        print(f"  most_captained: {sample_gw.get('most_captained'):,}")

if __name__ == "__main__":
    check_fpl_data()
