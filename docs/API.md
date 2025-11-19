# FPL AI Optimizer - API Documentation

Base URL: `http://localhost:8000`

## Authentication

Currently, no authentication is required. Future versions may include API key authentication.

## Endpoints

### Players

#### GET `/api/players`
Get all players with optional filtering.

**Query Parameters:**
- `position` (optional): Filter by position (GKP, DEF, MID, FWD)
- `team` (optional): Filter by team name
- `search` (optional): Search player names
- `min_price` (optional): Minimum price in millions
- `max_price` (optional): Maximum price in millions
- `min_ownership` (optional): Minimum ownership percentage
- `min_form` (optional): Minimum form rating

**Response:** Array of PlayerResponse objects

#### GET `/api/players/search`
Search players by name.

**Query Parameters:**
- `search` (required): Search term
- `position` (optional): Filter by position

**Response:** Array of PlayerResponse objects (max 20)

---

### Teams

#### GET `/api/teams`
Get all Premier League teams.

**Response:** Array of team objects

#### POST `/api/team/validate`
Validate team structure and constraints.

**Request Body:**
```json
{
  "player_ids": [1, 2, 3, ..., 15]
}
```

**Response:**
```json
{
  "is_valid": true,
  "errors": [],
  "warnings": [],
  "budget": {
    "total": 98.5,
    "remaining": 1.5
  },
  "structure": {
    "GKP": 2,
    "DEF": 5,
    "MID": 5,
    "FWD": 3
  },
  "team_counts": {
    "Arsenal": 2,
    "Manchester City": 3,
    ...
  }
}
```

---

### Analysis

#### POST `/api/analyze-team`
Analyze team and get transfer/captain suggestions.

**Request Body:**
```json
{
  "players": [1, 2, 3, ..., 15],
  "analyze_transfers": true,
  "analyze_captain": true
}
```

**Response:**
```json
{
  "team_analysis": {
    "total_value": 98.5,
    "total_points": 450,
    "average_form": 4.2,
    "position_breakdown": {...}
  },
  "transfer_suggestions": [...],
  "captain_suggestions": [...]
}
```

#### POST `/api/advanced-optimize`
Advanced team optimization with comprehensive analysis.

**Request Body:**
```json
{
  "players": [1, 2, 3, ..., 15],
  "budget": 100.0,
  "free_transfers": 1,
  "use_wildcard": false,
  "formation": "3-4-3",
  "preferences": {
    "risk_tolerance": 0.5,
    "formation_preference": "3-4-3",
    "differential_threshold": 5.0,
    "captain_strategy": "fixture_based",
    "transfer_frequency": "moderate"
  },
  "include_fixture_analysis": true,
  "include_price_analysis": true,
  "include_strategic_planning": true
}
```

**Response:** Comprehensive optimization results

---

### Managers

#### GET `/api/managers/search`
Search for FPL managers.

**Query Parameters:**
- `name` (required): Manager name to search

**Response:**
```json
{
  "managers": [
    {
      "id": "12345",
      "player_name": "John Doe",
      "team_name": "FC Awesome",
      "overall_rank": 50000,
      "total_points": 1250
    }
  ],
  "message": "Manager search results"
}
```

#### GET `/api/team/manager/{manager_id}`
Get manager's current team.

**Path Parameters:**
- `manager_id` (required): FPL manager ID

**Response:** Manager team with all players

---

### System

#### GET `/api/system/status`
Get system status.

**Response:**
```json
{
  "database_connected": true,
  "models_trained": true,
  "last_data_update": "2024-01-15 10:30:00",
  "current_gameweek": 20,
  "player_count": 650,
  "upcoming_fixtures": 40
}
```

#### GET `/api/system/data-status`
Check if player data is available.

**Response:**
```json
{
  "has_data": true,
  "player_count": 650,
  "message": "Data available"
}
```

#### POST `/api/system/refresh-data`
Refresh FPL data from API.

**Response:**
```json
{
  "success": true,
  "message": "Data updated successfully"
}
```

#### POST `/api/system/train-models`
Train AI models.

**Response:**
```json
{
  "success": true,
  "message": "Models trained successfully",
  "details": {
    "minutes_model": true,
    "points_model": true,
    "ensemble_model": true
  }
}
```

#### GET `/api/system/model-status`
Get AI model training status.

**Response:** Model status for all models

---

## Error Responses

All endpoints may return error responses:

```json
{
  "detail": "Error message here"
}
```

Common status codes:
- `400`: Bad Request
- `404`: Not Found
- `500`: Internal Server Error

## Interactive Documentation

Visit http://localhost:8000/api/docs for interactive Swagger UI documentation.

