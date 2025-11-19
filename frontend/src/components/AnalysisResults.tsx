'use client';

import { Player } from '@/lib/types';
import { formatPrice } from '@/lib/utils';

interface AnalysisResultsProps {
  team: Player[];
  analysisData: any;
  onReset: () => void;
}

export default function AnalysisResults({ team, analysisData, onReset }: AnalysisResultsProps) {
  if (!analysisData) {
    return (
      <div className="text-center py-12">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
        <p className="text-gray-600">Analyzing your team...</p>
      </div>
    );
  }

  const { team_analysis, transfer_suggestions, captain_suggestions } = analysisData;

  return (
    <div className="space-y-8">
      {/* Enhanced Team Overview */}
      <div className="glass rounded-2xl shadow-large border border-white/30 p-8">
        <div className="text-center mb-8">
          <h2 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-2">
            👥 Your Team Analysis
          </h2>
          <p className="text-gray-600">AI-powered insights and recommendations</p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="text-center bg-gradient-to-br from-blue-50 to-blue-100 rounded-2xl p-6 card-hover border border-blue-200/50">
            <div className="w-12 h-12 bg-blue-500 rounded-xl flex items-center justify-center mx-auto mb-3">
              <span className="text-white text-xl">💰</span>
            </div>
            <div className="text-3xl font-bold text-blue-600 mb-1">
              £{team_analysis.total_value?.toFixed(1)}M
            </div>
            <div className="text-sm font-medium text-blue-700">Team Value</div>
          </div>
          
          <div className="text-center bg-gradient-to-br from-green-50 to-green-100 rounded-2xl p-6 card-hover border border-green-200/50">
            <div className="w-12 h-12 bg-green-500 rounded-xl flex items-center justify-center mx-auto mb-3">
              <span className="text-white text-xl">🏆</span>
            </div>
            <div className="text-3xl font-bold text-green-600 mb-1">
              {team_analysis.total_points}
            </div>
            <div className="text-sm font-medium text-green-700">Total Points</div>
          </div>
          
          <div className="text-center bg-gradient-to-br from-purple-50 to-purple-100 rounded-2xl p-6 card-hover border border-purple-200/50">
            <div className="w-12 h-12 bg-purple-500 rounded-xl flex items-center justify-center mx-auto mb-3">
              <span className="text-white text-xl">📈</span>
            </div>
            <div className="text-3xl font-bold text-purple-600 mb-1">
              {team_analysis.average_form?.toFixed(1)}
            </div>
            <div className="text-sm font-medium text-purple-700">Avg Form</div>
          </div>
          
          <div className="text-center bg-gradient-to-br from-orange-50 to-orange-100 rounded-2xl p-6 card-hover border border-orange-200/50">
            <div className="w-12 h-12 bg-orange-500 rounded-xl flex items-center justify-center mx-auto mb-3">
              <span className="text-white text-xl">💳</span>
            </div>
            <div className="text-3xl font-bold text-orange-600 mb-1">
              £{(100 - team_analysis.total_value).toFixed(1)}M
            </div>
            <div className="text-sm font-medium text-orange-700">Budget Left</div>
          </div>
        </div>

        {/* Team Breakdown */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {['GKP', 'DEF', 'MID', 'FWD'].map(position => {
            const positionPlayers = team.filter(p => p.position === position);
            const positionIcons = { GKP: '🥅', DEF: '🛡️', MID: '⚽', FWD: '🎯' };
            
            return (
              <div key={position} className="bg-gray-50 rounded-lg p-4">
                <div className="flex items-center space-x-2 mb-3">
                  <span className="text-lg">{positionIcons[position as keyof typeof positionIcons]}</span>
                  <span className="font-medium text-gray-900">{position}</span>
                  <span className="text-sm text-gray-500">({positionPlayers.length})</span>
                </div>
                <div className="space-y-1">
                  {positionPlayers.map(player => (
                    <div key={player.id} className="text-sm">
                      <div className="font-medium text-gray-900">{player.web_name}</div>
                      <div className="text-gray-500">{formatPrice(player.now_cost)} • {player.total_points} pts</div>
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Captain Suggestions */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-6">⭐ Captain Recommendations</h2>
        
        {captain_suggestions && captain_suggestions.length > 0 ? (
          <div className="space-y-4">
            {captain_suggestions.slice(0, 3).map((suggestion: any, index: number) => (
              <div 
                key={index} 
                className={`p-4 rounded-lg border-2 transition-colors ${
                  index === 0 
                    ? 'border-yellow-300 bg-yellow-50' 
                    : 'border-gray-200 bg-gray-50'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    {index === 0 && <span className="text-xl">👑</span>}
                    <div>
                      <div className="font-semibold text-gray-900">
                        {suggestion.player_name}
                      </div>
                      <div className="text-sm text-gray-600">
                        {suggestion.reason}
                      </div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-lg font-bold text-yellow-600">
                      {suggestion.expected_points} pts
                    </div>
                    <div className="text-sm text-gray-500">
                      {suggestion.risk} risk
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8 text-gray-500">
            🤖 Calculating best captain options...
          </div>
        )}
      </div>

      {/* Transfer Suggestions */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-6">🔄 Transfer Suggestions</h2>
        
        {transfer_suggestions && transfer_suggestions.length > 0 ? (
          <div className="space-y-4">
            {transfer_suggestions.slice(0, 5).map((transfer: any, index: number) => (
              <div 
                key={index} 
                className={`border rounded-lg p-4 transition-colors ${
                  transfer.priority === 'High' 
                    ? 'border-green-300 bg-green-50' 
                    : 'border-gray-200 hover:border-blue-300'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-4 mb-2">
                      <span className="text-red-600 font-medium">❌ {transfer.player_out}</span>
                      <span className="text-gray-400">→</span>
                      <span className="text-green-600 font-medium">✅ {transfer.player_in}</span>
                    </div>
                    <div className="text-sm text-gray-600">{transfer.reason}</div>
                    <div className="text-xs text-gray-500 mt-1">
                      Priority: <span className="font-medium">{transfer.priority}</span>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-lg font-bold text-green-600">
                      +{transfer.points_gain} pts
                    </div>
                    <div className="text-sm text-gray-500">
                      {transfer.cost_change}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8">
            <div className="text-green-600 text-lg font-medium mb-2">✅ Your team looks optimized!</div>
            <p className="text-gray-500">No obvious transfer improvements found.</p>
          </div>
        )}
      </div>

      {/* Action Buttons */}
      <div className="flex justify-center space-x-4">
        <button
          onClick={onReset}
          className="px-6 py-3 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors font-medium"
        >
          🔄 Analyze Another Team
        </button>
        <button
          onClick={() => window.print()}
          className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-medium"
        >
          🖨️ Print Results
        </button>
      </div>
    </div>
  );
}
