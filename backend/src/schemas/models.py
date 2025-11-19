"""Pydantic models for API request/response validation"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class PlayerResponse(BaseModel):
    """Player data response model"""
    id: int
    web_name: str
    first_name: str
    second_name: str
    position: str
    team_name: str
    now_cost: int
    total_points: int
    form: float
    selected_by_percent: float
    points_per_game: float
    goals_scored: int
    assists: int
    clean_sheets: int
    saves: int
    bonus: int
    ict_index: float
    influence: float
    creativity: float
    threat: float
    status: str
    chance_of_playing_this_round: Optional[int] = None
    chance_of_playing_next_round: Optional[int] = None
    news: Optional[str] = None


class TeamValidationRequest(BaseModel):
    """Request model for team validation"""
    player_ids: List[int] = Field(..., min_length=1, max_length=15)


class TeamValidationResponse(BaseModel):
    """Response model for team validation"""
    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    budget: Dict[str, float]
    structure: Dict[str, int]
    team_counts: Dict[str, int]


class SystemStatusResponse(BaseModel):
    """System status response model"""
    database_connected: bool
    models_trained: bool
    last_data_update: Optional[str] = None
    current_gameweek: Optional[int] = None
    player_count: int
    upcoming_fixtures: int


class DataStatusResponse(BaseModel):
    """Data availability status response"""
    has_data: bool
    player_count: int
    message: str


class ManagerSearchResponse(BaseModel):
    """Manager search results response"""
    managers: List[Dict[str, Any]]
    message: str


class ManagerTeamResponse(BaseModel):
    """Manager team data response"""
    manager_id: str
    manager_name: str
    team_name: str
    overall_rank: Optional[int] = None
    total_points: Optional[int] = None
    players: List[Dict[str, Any]]


class TeamAnalysisRequest(BaseModel):
    """Request model for team analysis"""
    players: List[int] = Field(..., min_length=1, max_length=15)
    analyze_transfers: bool = True
    analyze_captain: bool = True


class TeamAnalysisResponse(BaseModel):
    """Response model for team analysis"""
    team_analysis: Dict[str, Any]
    transfer_suggestions: List[Dict[str, Any]]
    captain_suggestions: List[Dict[str, Any]]


class ManagerPreferencesModel(BaseModel):
    """Manager preferences for optimization"""
    risk_tolerance: float = Field(default=0.5, ge=0.0, le=1.0)
    formation_preference: str = "3-4-3"
    budget_allocation: Dict[str, float] = Field(default_factory=dict)
    differential_threshold: float = 5.0
    captain_strategy: str = "fixture_based"
    transfer_frequency: str = "moderate"


class AdvancedOptimizeRequest(BaseModel):
    """Request model for advanced optimization"""
    players: List[int] = Field(..., min_length=1, max_length=15)
    starting_xi: Optional[List[int]] = Field(default=None, max_length=11)
    budget: float = Field(default=100.0, ge=0.0, le=200.0)
    bank_amount: float = Field(default=0.0, ge=0.0, le=20.0)
    free_transfers: int = Field(default=1, ge=0, le=15)
    use_wildcard: bool = False
    chips_available: List[str] = Field(default_factory=lambda: ['Wildcard', 'Bench Boost', 'Triple Captain', 'Free Hit'])
    formation: str = "3-4-3"
    preferences: Optional[ManagerPreferencesModel] = None
    include_fixture_analysis: bool = True
    include_price_analysis: bool = True
    include_strategic_planning: bool = True


class AdvancedOptimizeResponse(BaseModel):
    """Response model for advanced optimization"""
    optimization_success: bool
    team_optimization: Dict[str, Any] = Field(default_factory=dict)
    team_analysis: Dict[str, Any] = Field(default_factory=dict)
    fixture_analysis: Dict[str, Any] = Field(default_factory=dict)
    price_analysis: Dict[str, Any] = Field(default_factory=dict)
    strategic_planning: Dict[str, Any] = Field(default_factory=dict)
    transfer_suggestions: List[Dict[str, Any]] = Field(default_factory=list)
    captain_suggestions: List[Dict[str, Any]] = Field(default_factory=list)
    chip_opportunities: List[Dict[str, Any]] = Field(default_factory=list)
    long_term_strategy: Dict[str, Any] = Field(default_factory=dict)
    confidence_score: float
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    features_used: Dict[str, Any] = Field(default_factory=dict)


class ModelStatusResponse(BaseModel):
    """Model training status response"""
    minutes_model: Dict[str, Any]
    points_model: Dict[str, Any]
    ensemble_model: Dict[str, Any]
    price_predictor: Dict[str, Any]
    fixture_analyzer: Dict[str, Any]
    strategic_planner: Dict[str, Any]


class RefreshDataResponse(BaseModel):
    """Data refresh response"""
    success: bool
    message: str


class TrainModelsResponse(BaseModel):
    """Model training response"""
    success: bool
    message: str
    details: Dict[str, bool] = Field(default_factory=dict)


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str
    message: str

