'use client';

import { useEffect, useState } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import { Player, AdvancedOptimizationResult, ManagerPreferences } from '@/lib/types';
import { formatPrice, getPositionIcon, getTeamColor } from '@/lib/utils';
import { FPLApi } from '@/lib/api';

interface TeamConfig {
  free_transfers: number;
  bank: number;
  chips_available: string[];
  starting_xi: number[];
}

export default function AdvancedAnalysisPage() {
  const [team, setTeam] = useState<Player[]>([]);
  const [analysisData, setAnalysisData] = useState<AdvancedOptimizationResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [preferences, setPreferences] = useState<ManagerPreferences>({
    risk_tolerance: 0.5,
    formation_preference: '3-4-3',
    budget_allocation: { GKP: 0.08, DEF: 0.25, MID: 0.40, FWD: 0.27 },
    differential_threshold: 5.0,
    captain_strategy: 'fixture_based',
    transfer_frequency: 'moderate'
  });
  const [analysisOptions, setAnalysisOptions] = useState({
    include_fixture_analysis: true,
    include_price_analysis: true,
    include_strategic_planning: true
  });
  const [teamConfig, setTeamConfig] = useState<TeamConfig>({
    free_transfers: 1,
    bank: 0.0,
    chips_available: ['Wildcard', 'Bench Boost', 'Triple Captain', 'Free Hit'],
    starting_xi: []
  });
  const [activeTab, setActiveTab] = useState<'overview' | 'transfers' | 'chips' | 'bench' | 'advanced'>('overview');
  
  const searchParams = useSearchParams();
  const router = useRouter();

  useEffect(() => {
    const teamData = searchParams.get('team');
    const configData = searchParams.get('config');
    
    if (!teamData) {
      router.push('/');
      return;
    }

    try {
      const parsedTeam = JSON.parse(decodeURIComponent(teamData));
      setTeam(parsedTeam);
      
      // Parse config if available
      let effectiveConfig: TeamConfig = {
        free_transfers: 1,
        bank: 0.0,
        chips_available: ['Wildcard', 'Bench Boost', 'Triple Captain', 'Free Hit'],
        starting_xi: []
      };
      if (configData) {
        try {
          const parsedConfig = JSON.parse(decodeURIComponent(configData));
          effectiveConfig = {
            free_transfers: parsedConfig.free_transfers ?? 1,
            bank: parsedConfig.bank ?? parsedConfig.budget ?? 0.0,
            chips_available: parsedConfig.chips_available ?? ['Wildcard', 'Bench Boost', 'Triple Captain', 'Free Hit'],
            starting_xi: Array.isArray(parsedConfig.starting_xi) ? parsedConfig.starting_xi : []
          };
        } catch (e) {
          console.error('Error parsing config:', e);
        }
      }
      setTeamConfig(effectiveConfig);
      
      // Fetch advanced analysis data with the parsed config
      fetchAdvancedAnalysis(parsedTeam, effectiveConfig);
    } catch (error) {
      console.error('Error parsing team data:', error);
      router.push('/');
    }
  }, [searchParams, router]);

  const calculateTeamValue = (players: Player[]) =>
    players.reduce((sum, player) => sum + (player?.now_cost || 0) / 10, 0);

  const fetchAdvancedAnalysis = async (teamPlayers: Player[], configOverride?: TeamConfig) => {
    try {
      setLoading(true);
      const effectiveConfig = configOverride ?? teamConfig;
      const totalTeamValue = calculateTeamValue(teamPlayers);
      const totalBudget = Number((totalTeamValue + (effectiveConfig.bank ?? 0)).toFixed(1));
      const result = await FPLApi.advancedOptimizeTeam({
        players: teamPlayers.map(p => p.id),
        budget: totalBudget,
        free_transfers: effectiveConfig.free_transfers,
        use_wildcard: false,
        chips_available: effectiveConfig.chips_available,
        starting_xi: effectiveConfig.starting_xi,
        bank_amount: effectiveConfig.bank ?? 0,
        formation: '3-4-3',
        preferences: preferences,
        ...analysisOptions
      });
      
      setAnalysisData(result);
    } catch (error) {
      console.error('Advanced analysis failed:', error);
      setError('Failed to analyze team. Please try again.');
    } finally {
      setLoading(false);
    }
  };
  
  const handleReanalyze = () => {
    fetchAdvancedAnalysis(team, teamConfig);
  };

  const getConfidenceColor = (score: number) => {
    if (score >= 0.8) return 'text-green-600 bg-green-100';
    if (score >= 0.6) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'High': return 'bg-red-100 text-red-800 border-red-200';
      case 'Medium': return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'Low': return 'bg-green-100 text-green-800 border-green-200';
      default: return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-4 border-blue-600 mx-auto mb-4"></div>
          <h2 className="text-2xl font-bold text-gray-900 mb-2">🤖 Advanced AI Analysis</h2>
          <p className="text-gray-600">Running comprehensive optimization with all features...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 flex items-center justify-center">
        <div className="text-center">
          <div className="text-6xl mb-4">❌</div>
          <h2 className="text-2xl font-bold text-gray-900 mb-2">Analysis Failed</h2>
          <p className="text-gray-600 mb-4">{error}</p>
          <button
            onClick={() => router.push('/')}
            className="px-6 py-3 bg-blue-600 text-white rounded-xl hover:bg-blue-700 transition-colors"
          >
            🔙 Back to Team Selection
          </button>
        </div>
      </div>
    );
  }

  const benchData = analysisData?.bench_analysis || null;
  const allChips = ['Wildcard', 'Bench Boost', 'Triple Captain', 'Free Hit'];
  const availableChips = analysisData?.team_analysis?.chips_available ?? teamConfig.chips_available;
  const chipsAvailableSet = new Set(availableChips);
  const chipOpportunities = (analysisData?.chip_opportunities || []).filter(
    (chip: any) => chipsAvailableSet.has(chip.chip)
  );
  const predictedPointsValue = analysisData?.team_analysis?.predicted_points;
  const predictedPointsDisplay = typeof predictedPointsValue === 'number' ? predictedPointsValue.toFixed(1) : '0.0';
  const teamStrengthValue = analysisData?.team_analysis?.team_strength;
  const teamStrengthDisplay = typeof teamStrengthValue === 'number' ? teamStrengthValue.toFixed(1) : '0.0';
  const formationDisplay = analysisData?.team_analysis?.formation || '3-4-3';
  const freeTransfersDisplay = analysisData?.team_analysis?.free_transfers ?? teamConfig.free_transfers;
  const bankValue =
    typeof analysisData?.team_analysis?.bank_remaining === 'number'
      ? analysisData.team_analysis.bank_remaining
      : teamConfig.bank ?? 0;
  const bankDisplay = bankValue.toFixed(1);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50">
      {/* Enhanced Header */}
      <div className="bg-white/80 backdrop-blur-sm border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <button
                onClick={() => router.push('/')}
                className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
              >
                ← Back
              </button>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">🚀 Advanced FPL Analysis</h1>
                <p className="text-gray-600 text-sm">AI-powered comprehensive team optimization</p>
              </div>
            </div>
            
            {/* Confidence Score */}
            {analysisData ? (
              <div className="text-right">
                <div className="text-sm text-gray-600">AI Confidence</div>
                <div className={`text-xl font-bold px-3 py-1 rounded-full ${getConfidenceColor(analysisData.confidence_score)}`}>
                  {(analysisData.confidence_score * 100).toFixed(0)}%
                </div>
                <div className="text-xs text-gray-500">
                  {analysisData.features_used?.total_features || 0}/6 features active
                </div>
              </div>
            ) : (
              <div className="text-right text-sm text-gray-500">
                AI Confidence pending...
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          
          {/* Sidebar - Team Overview & Settings */}
          <div className="lg:col-span-1 space-y-6">
            
            {/* Team Overview */}
            <div className="bg-white rounded-2xl shadow-lg p-6 border border-gray-200">
              <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center">
                👥 Team Overview
              </h3>
              
              <div className="space-y-4">
                <div className="flex justify-between items-center p-3 bg-blue-50 rounded-lg">
                  <span className="text-gray-700">Formation</span>
                  <span className="font-bold text-blue-600">{formationDisplay}</span>
                </div>
                
                <div className="flex justify-between items-center p-3 bg-green-50 rounded-lg">
                  <span className="text-gray-700">Predicted Points</span>
                  <span className="font-bold text-green-600">
                    {predictedPointsDisplay} pts
                  </span>
                </div>
                
                <div className="flex justify-between items-center p-3 bg-purple-50 rounded-lg">
                  <span className="text-gray-700">Team Strength</span>
                  <span className="font-bold text-purple-600">
                    {teamStrengthDisplay}/10
                  </span>
                </div>
                
                <div className="flex justify-between items-center p-3 bg-yellow-50 rounded-lg">
                  <span className="text-gray-700">Free Transfers</span>
                  <span className="font-bold text-yellow-600">{freeTransfersDisplay}</span>
                </div>

                <div className="flex justify-between items-center p-3 bg-indigo-50 rounded-lg">
                  <span className="text-gray-700">Bank</span>
                  <span className="font-bold text-indigo-600">
                    £{bankDisplay}m
                  </span>
                </div>
              </div>

              {/* Chips Status */}
              <div className="mt-6">
                <h4 className="font-semibold text-gray-900 mb-3">🎯 Chips Status</h4>
                <div className="grid grid-cols-2 gap-2">
                  {allChips.map((chip) => {
                    const isAvailable = chipsAvailableSet.has(chip);
                    return (
                      <div
                        key={chip}
                        className={`p-2 rounded-lg text-xs font-medium text-center ${
                          isAvailable
                            ? 'bg-green-100 text-green-700'
                            : 'bg-gray-100 text-gray-500'
                        }`}
                      >
                        {chip} {isAvailable ? '🎯' : '✅'}
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Analysis Settings */}
            <div className="bg-white rounded-2xl shadow-lg p-6 border border-gray-200">
              <h3 className="text-xl font-bold text-gray-900 mb-4">⚙️ Analysis Settings</h3>
              
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-700">Fixture Analysis</span>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={analysisOptions.include_fixture_analysis}
                      onChange={(e) => setAnalysisOptions(prev => ({
                        ...prev,
                        include_fixture_analysis: e.target.checked
                      }))}
                      className="sr-only peer"
                    />
                    <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                  </label>
                </div>
                
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-700">Price Analysis</span>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={analysisOptions.include_price_analysis}
                      onChange={(e) => setAnalysisOptions(prev => ({
                        ...prev,
                        include_price_analysis: e.target.checked
                      }))}
                      className="sr-only peer"
                    />
                    <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                  </label>
                </div>
                
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-700">Strategic Planning</span>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={analysisOptions.include_strategic_planning}
                      onChange={(e) => setAnalysisOptions(prev => ({
                        ...prev,
                        include_strategic_planning: e.target.checked
                      }))}
                      className="sr-only peer"
                    />
                    <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                  </label>
                </div>
              </div>
              
              <button
                onClick={() => fetchAdvancedAnalysis(team)}
                className="w-full mt-4 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                🔄 Refresh Analysis
              </button>
            </div>
          </div>

          {/* Main Content */}
          <div className="lg:col-span-3 space-y-6">
            
            {/* Tab Navigation */}
            <div className="bg-white rounded-2xl shadow-lg border border-gray-200 overflow-hidden">
              <div className="flex border-b border-gray-200 overflow-x-auto">
                <button
                  onClick={() => setActiveTab('overview')}
                  className={`px-6 py-4 font-medium text-sm whitespace-nowrap transition-colors ${
                    activeTab === 'overview'
                      ? 'bg-blue-50 text-blue-600 border-b-2 border-blue-600'
                      : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                  }`}
                >
                  📊 Overview
                </button>
                <button
                  onClick={() => setActiveTab('transfers')}
                  className={`px-6 py-4 font-medium text-sm whitespace-nowrap transition-colors ${
                    activeTab === 'transfers'
                      ? 'bg-blue-50 text-blue-600 border-b-2 border-blue-600'
                      : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                  }`}
                >
                  🔄 Transfers {analysisData?.transfer_suggestions?.length > 0 && `(${analysisData.transfer_suggestions.length})`}
                </button>
                <button
                  onClick={() => setActiveTab('chips')}
                  className={`px-6 py-4 font-medium text-sm whitespace-nowrap transition-colors ${
                    activeTab === 'chips'
                      ? 'bg-blue-50 text-blue-600 border-b-2 border-blue-600'
                      : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                  }`}
                >
                  🎯 Captain & Chips
                </button>
                <button
                  onClick={() => setActiveTab('bench')}
                  className={`px-6 py-4 font-medium text-sm whitespace-nowrap transition-colors ${
                    activeTab === 'bench'
                      ? 'bg-blue-50 text-blue-600 border-b-2 border-blue-600'
                      : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                  }`}
                >
                  ⚽ Bench & Formation
                </button>
                <button
                  onClick={() => setActiveTab('advanced')}
                  className={`px-6 py-4 font-medium text-sm whitespace-nowrap transition-colors ${
                    activeTab === 'advanced'
                      ? 'bg-blue-50 text-blue-600 border-b-2 border-blue-600'
                      : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                  }`}
                >
                  🔬 Advanced
                </button>
              </div>
            </div>

            {/* Overview Tab */}
            {activeTab === 'overview' && (
              <div className="space-y-6">
                {/* Team Summary */}
                <div className="bg-white rounded-2xl shadow-lg p-6 border border-gray-200">
                  <h3 className="text-xl font-bold text-gray-900 mb-4">📊 Team Summary</h3>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="p-4 bg-blue-50 rounded-lg">
                      <div className="text-sm text-gray-600 mb-1">Team Value</div>
                      <div className="text-2xl font-bold text-blue-600">
                        £{analysisData?.team_analysis?.total_value || '0.0'}M
                      </div>
                    </div>
                    <div className="p-4 bg-green-50 rounded-lg">
                      <div className="text-sm text-gray-600 mb-1">Predicted Points</div>
                      <div className="text-2xl font-bold text-green-600">
                        {analysisData?.team_analysis?.predicted_points || '0.0'}
                      </div>
                    </div>
                    <div className="p-4 bg-purple-50 rounded-lg">
                      <div className="text-sm text-gray-600 mb-1">Team Strength</div>
                      <div className="text-2xl font-bold text-purple-600">
                        {analysisData?.team_analysis?.team_strength || '0.0'}/10
                      </div>
                    </div>
                    <div className="p-4 bg-yellow-50 rounded-lg">
                      <div className="text-sm text-gray-600 mb-1">Free Transfers</div>
                      <div className="text-2xl font-bold text-yellow-600">
                        {analysisData?.team_analysis?.free_transfers || 1}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Quick Actions */}
                <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-2xl p-6 border border-blue-200">
                  <h3 className="text-xl font-bold text-gray-900 mb-4">🎯 Quick Actions</h3>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <button
                      onClick={() => setActiveTab('transfers')}
                      className="p-4 bg-white rounded-lg hover:shadow-lg transition-shadow text-left"
                    >
                      <div className="text-2xl mb-2">🔄</div>
                      <div className="font-semibold text-gray-900">View Transfers</div>
                      <div className="text-sm text-gray-600">
                        {analysisData?.transfer_suggestions?.length || 0} suggestions
                      </div>
                    </button>
                    <button
                      onClick={() => setActiveTab('chips')}
                      className="p-4 bg-white rounded-lg hover:shadow-lg transition-shadow text-left"
                    >
                      <div className="text-2xl mb-2">👑</div>
                      <div className="font-semibold text-gray-900">Captain & Chips</div>
                      <div className="text-sm text-gray-600">
                        {analysisData?.captain_suggestions?.length || 0} options
                      </div>
                    </button>
                    <button
                      onClick={() => setActiveTab('bench')}
                      className="p-4 bg-white rounded-lg hover:shadow-lg transition-shadow text-left"
                    >
                      <div className="text-2xl mb-2">⚽</div>
                      <div className="font-semibold text-gray-900">Optimize Bench</div>
                      <div className="text-sm text-gray-600">
                        Strength: {analysisData?.bench_analysis?.bench_strength || '0.0'}/10
                      </div>
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* Transfers Tab */}
            {activeTab === 'transfers' && (
              <div className="bg-white rounded-2xl shadow-lg p-6 border border-gray-200">
              <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center">
                🔄 Transfer Recommendations
              </h3>
              
              <div className="space-y-4">
                {analysisData?.transfer_suggestions?.length > 0 ? (
                  analysisData.transfer_suggestions.map((transfer, index) => (
                    <div key={index} className={`p-4 rounded-xl border-2 ${
                      transfer.is_free 
                        ? 'bg-gradient-to-r from-green-50 to-emerald-50 border-green-300' 
                        : 'bg-gradient-to-r from-orange-50 to-red-50 border-orange-300'
                    }`}>
                      <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center space-x-2">
                          <div className={`text-xs px-3 py-1 rounded-full font-medium border ${getPriorityColor(transfer.priority)}`}>
                            {transfer.priority} Priority
                          </div>
                          {transfer.is_free ? (
                            <div className="text-xs px-3 py-1 rounded-full font-medium bg-green-100 text-green-800 border border-green-200">
                              ✅ Free Transfer
                            </div>
                          ) : transfer.cost_warning && (
                            <div className="text-xs px-3 py-1 rounded-full font-medium bg-red-100 text-red-800 border border-red-200">
                              ⚠️ {transfer.cost_warning}
                            </div>
                          )}
                        </div>
                        <div className="text-right">
                          <div className="text-lg font-bold text-green-600">
                            +{transfer.points_gain.toFixed(1)} pts
                          </div>
                        </div>
                      </div>
                      
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-center">
                        <div className="text-center">
                          <div className="text-xs text-red-600 font-medium mb-2 flex items-center justify-center space-x-1">
                            <span>❌</span>
                            <span>TRANSFER OUT</span>
                          </div>
                          <div className="p-4 bg-white rounded-xl shadow-sm border-2 border-red-200">
                            <div className="mb-3">
                              <div className="w-16 h-16 mx-auto bg-gradient-to-br from-gray-100 to-gray-200 rounded-full flex items-center justify-center text-2xl">
                                {getPositionIcon(transfer.player_out.position)}
                              </div>
                            </div>
                            <div className="font-bold text-gray-900 text-sm mb-1">{transfer.player_out.web_name}</div>
                            <div className="text-xs text-gray-600 mb-2">{transfer.player_out.team_name}</div>
                          </div>
                        </div>
                        
                        <div className="text-center flex flex-col items-center justify-center">
                          <div className="text-3xl text-blue-600 mb-2">→</div>
                        </div>
                        
                        <div className="text-center">
                          <div className="text-xs text-green-600 font-medium mb-2 flex items-center justify-center space-x-1">
                            <span>✅</span>
                            <span>TRANSFER IN</span>
                          </div>
                          <div className="p-4 bg-white rounded-xl shadow-sm border-2 border-green-200">
                            <div className="mb-3">
                              <div className="w-16 h-16 mx-auto bg-gradient-to-br from-green-100 to-emerald-200 rounded-full flex items-center justify-center text-2xl">
                                {getPositionIcon(transfer.player_in.position)}
                              </div>
                            </div>
                            <div className="font-bold text-gray-900 text-sm mb-1">{transfer.player_in.web_name}</div>
                            <div className="text-xs text-gray-600 mb-2">{transfer.player_in.team_name}</div>
                          </div>
                        </div>
                      </div>
                      
                      <div className="mt-4 space-y-3">
                        {/* Bench Guidance */}
                        {transfer.bench_guidance && (
                          <div className={`text-sm p-3 rounded-lg border ${
                            transfer.should_start 
                              ? 'bg-green-50 border-green-200' 
                              : 'bg-gray-50 border-gray-200'
                          }`}>
                            <div className="flex items-center space-x-2">
                              <span className={`text-lg ${transfer.should_start ? 'text-green-600' : 'text-gray-600'}`}>
                                {transfer.should_start ? '⚡' : '🪑'}
                              </span>
                              <div>
                                <strong className="text-gray-900">Lineup:</strong>
                                <span className={`ml-2 ${transfer.should_start ? 'text-green-700 font-medium' : 'text-gray-700'}`}>
                                  {transfer.bench_guidance}
                                </span>
                              </div>
                            </div>
                          </div>
                        )}
                        
                        {/* Reason */}
                        <div className="text-sm text-gray-700 bg-white/90 p-3 rounded-lg border border-gray-200">
                          <div className="flex items-start space-x-2">
                            <span className="text-blue-600 flex-shrink-0">💡</span>
                            <div>
                              <strong className="text-gray-900">Reason:</strong>
                              <span className="ml-2">{transfer.reason}</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-center py-8 text-gray-500">
                    <div className="text-4xl mb-4">🔄</div>
                    <h4 className="text-lg font-medium text-gray-700 mb-2">No Transfer Suggestions</h4>
                    <p className="text-sm text-gray-500">Your team looks good for now!</p>
                  </div>
                )}
              </div>
            </div>
            )}

            {/* Captain & Chips Tab */}
            {activeTab === 'chips' && (
              <div className="space-y-6">
                {/* Captain Suggestions */}
                <div className="bg-white rounded-2xl shadow-lg p-6 border border-gray-200">
                  <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center">
                    👑 Captain Recommendations
                  </h3>
                  
                  <div className="space-y-3">
                    {analysisData?.captain_suggestions?.length > 0 ? (
                      analysisData.captain_suggestions.map((suggestion, index) => (
                        <div key={index} className="p-4 bg-gradient-to-r from-yellow-50 to-orange-50 rounded-xl border border-yellow-200">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center space-x-3">
                              <div className="w-12 h-12 bg-yellow-100 rounded-full flex items-center justify-center">
                                <span className="text-lg">{getPositionIcon(suggestion.player.position)}</span>
                              </div>
                              <div>
                                <h4 className="font-bold text-gray-900">{suggestion.player.web_name}</h4>
                                <p className="text-sm text-gray-600">{suggestion.player.team_name}</p>
                              </div>
                            </div>
                            <div className="text-right">
                              <div className="text-lg font-bold text-green-600">
                                {suggestion.expected_points.toFixed(1)} pts
                              </div>
                              <div className={`text-xs px-2 py-1 rounded-full ${
                                suggestion.confidence === 'High' ? 'bg-green-100 text-green-800' :
                                suggestion.confidence === 'Medium' ? 'bg-yellow-100 text-yellow-800' :
                                'bg-red-100 text-red-800'
                              }`}>
                                {suggestion.confidence} confidence
                              </div>
                            </div>
                          </div>
                          <div className="mt-3 text-sm text-gray-700 bg-white/50 p-3 rounded-lg">
                            <strong>Why:</strong> {suggestion.reason}
                          </div>
                        </div>
                      ))
                    ) : (
                      <div className="text-center py-8 text-gray-500">
                        <div className="text-4xl mb-4">👑</div>
                        <h4 className="text-lg font-medium text-gray-700 mb-2">No Captain Suggestions</h4>
                        <p className="text-sm text-gray-500">Analysis in progress...</p>
                      </div>
                    )}
                  </div>
                </div>

                {/* Chip Recommendations */}
                {chipOpportunities.length > 0 && (
                  <div className="bg-white rounded-2xl shadow-lg p-6 border border-gray-200">
                    <h3 className="text-xl font-bold text-gray-900 mb-4">🎯 Chip Strategy</h3>
                    <div className="space-y-4">
                      {chipOpportunities.map((chip: any, index: number) => (
                        <div key={index} className={`p-4 rounded-xl border-2 ${
                          chip.score >= 7 ? 'bg-gradient-to-r from-green-50 to-emerald-50 border-green-300' :
                          chip.score >= 5 ? 'bg-gradient-to-r from-yellow-50 to-orange-50 border-yellow-300' :
                          'bg-gradient-to-r from-gray-50 to-slate-50 border-gray-300'
                        }`}>
                          <div className="flex items-center justify-between mb-3">
                            <div className="flex items-center space-x-3">
                              <div className={`w-12 h-12 rounded-full flex items-center justify-center text-xl ${
                                chip.score >= 7 ? 'bg-green-200' :
                                chip.score >= 5 ? 'bg-yellow-200' :
                                'bg-gray-200'
                              }`}>
                                {chip.chip === 'Triple Captain' ? '3️⃣' :
                                 chip.chip === 'Bench Boost' ? '⚡' :
                                 chip.chip === 'Wildcard' ? '🃏' :
                                 chip.chip === 'Free Hit' ? '🎯' : '🎲'}
                              </div>
                              <div>
                                <h4 className="font-bold text-gray-900">{chip.chip}</h4>
                              <p className="text-sm text-gray-600">
                                Recommended: {chip.recommended_gameweek || 'Upcoming GW'}
                              </p>
                              </div>
                            </div>
                            <div className="text-right">
                              <div className="text-2xl font-bold text-purple-600">
                                {chip.score.toFixed(1)}/10
                              </div>
                              <div className={`text-xs px-2 py-1 rounded-full ${
                                chip.confidence === 'High' ? 'bg-green-100 text-green-800' :
                                chip.confidence === 'Medium' ? 'bg-yellow-100 text-yellow-800' :
                                'bg-red-100 text-red-800'
                              }`}>
                                {chip.confidence}
                              </div>
                            </div>
                          </div>
                          <p className="text-sm text-gray-700 mb-3">{chip.reason}</p>
                          {chip.conditions && chip.conditions.length > 0 && (
                            <div className="bg-white/80 p-3 rounded-lg">
                              <div className="text-xs font-semibold text-gray-700 mb-2">Conditions:</div>
                              <ul className="text-xs text-gray-600 space-y-1">
                                {chip.conditions.map((condition: string, i: number) => (
                                  <li key={i}>• {condition}</li>
                                ))}
                              </ul>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Bench & Formation Tab */}
            {activeTab === 'bench' && analysisData && (
              <div className="bg-white rounded-2xl shadow-lg p-6 border border-gray-200">
                <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center">
                  ⚽ Bench & Formation Optimization
                </h3>
                {benchData && Object.keys(benchData).length > 0 ? (
                  <>
                    <div className="grid grid-cols-2 gap-4 mb-6">
                      <div className="p-4 bg-purple-50 rounded-lg border border-purple-200">
                        <div className="text-sm text-gray-600 mb-1">Bench Strength</div>
                        <div className="text-2xl font-bold text-purple-600">
                          {benchData?.bench_strength || 0}/10
                        </div>
                      </div>
                      <div className="p-4 bg-green-50 rounded-lg border border-green-200">
                        <div className="text-sm text-gray-600 mb-1">Bench Value</div>
                        <div className="text-2xl font-bold text-green-600">
                          £{benchData?.bench_value?.toFixed(1) || '0.0'}m
                        </div>
                      </div>
                    </div>
                    
                    {benchData?.autosub_potential?.length > 0 && (
                      <div className="mb-6">
                        <h4 className="font-semibold text-gray-900 mb-3">🔄 Autosub Recommendations</h4>
                        <div className="space-y-3">
                          {benchData.autosub_potential.map((sub: any, index: number) => (
                            <div key={index} className="p-3 bg-orange-50 rounded-lg border border-orange-200">
                              <div className="flex items-center justify-between mb-2">
                                <div className="flex items-center space-x-2">
                                  <span className="text-green-600 font-semibold">{sub.bench_player.web_name}</span>
                                  <span className="text-gray-400">→</span>
                                  <span className="text-red-600 font-semibold">{sub.starter_to_replace.web_name}</span>
                                </div>
                                <div className="text-green-600 font-bold">+{sub.points_gain} pts</div>
                              </div>
                              <p className="text-xs text-gray-600">{sub.recommendation}</p>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                    
                    {benchData?.bench_players?.length > 0 && (
                      <div className="mb-6">
                        <h4 className="font-semibold text-gray-900 mb-3">🪑 Bench Players</h4>
                        <div className="grid grid-cols-2 gap-2">
                          {benchData.bench_players.map((player: any, index: number) => (
                            <div key={index} className="p-2 bg-gray-50 rounded-lg text-xs">
                              <div className="font-semibold text-gray-900">{player.web_name}</div>
                              <div className="text-gray-600">{player.team_name} • {player.position}</div>
                              <div className="text-gray-500">{player.predicted_points} pts • £{player.cost}m</div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                    
                    {benchData?.recommendations?.length > 0 && (
                      <div>
                        <h4 className="font-semibold text-gray-900 mb-3">💡 Recommendations</h4>
                        <div className="space-y-2">
                          {benchData.recommendations.map((rec: any, index: number) => (
                            <div key={index} className={`p-3 rounded-lg border ${
                              rec.type === 'warning' ? 'bg-yellow-50 border-yellow-200' :
                              rec.type === 'chip' ? 'bg-purple-50 border-purple-200' :
                              rec.type === 'action' ? 'bg-red-50 border-red-200' :
                              'bg-blue-50 border-blue-200'
                            }`}>
                              <div className="flex items-start justify-between">
                                <p className="text-sm text-gray-700 flex-1">{rec.message}</p>
                                <span className={`text-xs px-2 py-1 rounded-full ml-2 whitespace-nowrap ${
                                  rec.priority === 'High' ? 'bg-red-100 text-red-800' :
                                  rec.priority === 'Medium' ? 'bg-yellow-100 text-yellow-800' :
                                  'bg-blue-100 text-blue-800'
                                }`}>
                                  {rec.priority}
                                </span>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </>
                ) : (
                  <div className="text-center text-gray-600 py-10">
                    <p className="text-lg font-semibold mb-2">No bench data yet</p>
                    <p className="text-sm">Select and submit your starting XI to unlock bench insights.</p>
                  </div>
                )}
              </div>
            )}

            {/* Advanced Tab */}
            {activeTab === 'advanced' && analysisData && (
              <div className="bg-white rounded-2xl shadow-lg p-6 border border-gray-200">
                <h3 className="text-xl font-bold text-gray-900 mb-4">🔬 Advanced Analysis</h3>
                
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  
                  {/* Fixture Analysis */}
                  {analysisData.fixture_analysis && (
                    <div className="p-4 bg-blue-50 rounded-lg border border-blue-200">
                      <h4 className="font-semibold text-blue-900 mb-2">📅 Fixture Analysis</h4>
                      <p className="text-sm text-blue-700">
                        {analysisData.fixture_analysis.double_gameweeks?.length || 0} double gameweeks detected
                      </p>
                      <p className="text-sm text-blue-700">
                        {analysisData.fixture_analysis.blank_gameweeks?.length || 0} blank gameweeks detected
                      </p>
                    </div>
                  )}

                  {/* Price Analysis */}
                  {analysisData.price_analysis && (
                    <div className="p-4 bg-green-50 rounded-lg border border-green-200">
                      <h4 className="font-semibold text-green-900 mb-2">💰 Price Analysis</h4>
                      <p className="text-sm text-green-700">
                        {analysisData.price_analysis.rising_players?.length || 0} players rising
                      </p>
                      <p className="text-sm text-green-700">
                        {analysisData.price_analysis.falling_players?.length || 0} players falling
                      </p>
                    </div>
                  )}

                  {/* Strategic Planning */}
                  {analysisData.strategic_planning && (
                    <div className="p-4 bg-purple-50 rounded-lg border border-purple-200">
                      <h4 className="font-semibold text-purple-900 mb-2">🎯 Strategic Planning</h4>
                      <p className="text-sm text-purple-700">
                        {analysisData.strategic_planning.chip_opportunities?.length || 0} chip opportunities
                      </p>
                      <p className="text-sm text-purple-700">
                        Long-term strategy available
                      </p>
                    </div>
                  )}

                  {/* News Analysis */}
                  {analysisData.news_analysis && (
                    <div className="p-4 bg-orange-50 rounded-lg border border-orange-200">
                      <h4 className="font-semibold text-orange-900 mb-2">📰 News Analysis</h4>
                      <p className="text-sm text-orange-700">
                        {analysisData.news_analysis.high_risk_players?.length || 0} high-risk players
                      </p>
                      <p className="text-sm text-orange-700">
                        {analysisData.news_analysis.total_analyzed || 0} players analyzed
                      </p>
                    </div>
                  )}

                  {/* Effective Ownership */}
                  {analysisData.effective_ownership && (
                    <div className="p-4 bg-indigo-50 rounded-lg border border-indigo-200">
                      <h4 className="font-semibold text-indigo-900 mb-2">👥 Effective Ownership</h4>
                      <p className="text-sm text-indigo-700">
                        {analysisData.effective_ownership.template_team?.length || 0} template players
                      </p>
                      <p className="text-sm text-indigo-700">
                        {analysisData.effective_ownership.differential_opportunities?.length || 0} differentials
                      </p>
                    </div>
                  )}

                  {/* Model Status */}
                  <div className="p-4 bg-gray-50 rounded-lg border border-gray-200">
                    <h4 className="font-semibold text-gray-900 mb-2">🤖 AI Models</h4>
                    <p className="text-sm text-gray-700">
                      Ensemble: {analysisData.features_used?.ensemble_models ? '✅' : '❌'}
                    </p>
                    <p className="text-sm text-gray-700">
                      Features: {analysisData.features_used?.total_features || 0}/6 active
                    </p>
                  </div>
                </div>
                
                {/* Warnings in Advanced Tab */}
                {analysisData?.warnings && analysisData.warnings.length > 0 && (
                  <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-4 mt-6">
                    <h4 className="text-lg font-bold text-yellow-900 mb-3">⚠️ Analysis Warnings</h4>
                    <div className="space-y-2">
                      {analysisData.warnings.map((warning, index) => (
                        <div key={index} className="text-sm text-yellow-800 bg-yellow-100 p-2 rounded">
                          {warning}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Action Buttons - Visible on all tabs */}
            <div className="bg-white rounded-2xl shadow-lg p-6 border border-gray-200">
              <h3 className="text-xl font-bold text-gray-900 mb-4">🎯 Next Steps</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <button
                  onClick={() => window.open('https://fantasy.premierleague.com', '_blank')}
                  className="p-4 bg-gradient-to-r from-green-500 to-green-600 text-white rounded-xl hover:from-green-600 hover:to-green-700 transition-all duration-300 font-medium"
                >
                  🚀 Apply Changes in FPL
                </button>
                <button
                  onClick={() => fetchAdvancedAnalysis(team)}
                  className="p-4 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-xl hover:from-blue-600 hover:to-blue-700 transition-all duration-300 font-medium"
                >
                  🔄 Refresh Analysis
                </button>
                <button
                  onClick={() => router.push('/')}
                  className="p-4 bg-gradient-to-r from-gray-500 to-gray-600 text-white rounded-xl hover:from-gray-600 hover:to-gray-700 transition-all duration-300 font-medium"
                >
                  📝 New Team
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
