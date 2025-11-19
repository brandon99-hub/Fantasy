'use client';

import { useState } from 'react';
import dynamic from 'next/dynamic';
import { Player } from '@/lib/types';
import { validateTeamStructure } from '@/lib/utils';

// Lazy load FormationLayout for better performance
const FormationLayout = dynamic(() => import('./FormationLayout'), {
  loading: () => (
    <div className="flex items-center justify-center py-40">
      <div className="text-center">
        <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600 mx-auto mb-4"></div>
        <p className="text-gray-600">Loading pitch...</p>
      </div>
    </div>
  ),
  ssr: false
});

interface TeamUploadProps {
  onTeamUpload: (data: { players: Player[], config: any }) => void;
  loading: boolean;
}

export default function TeamUpload({ onTeamUpload, loading }: TeamUploadProps) {
  const [uploadMethod, setUploadMethod] = useState<'formation' | 'manager'>('formation');

  return (
    <div className="space-y-8">
      {/* Enhanced Upload Method Selection */}
      <div className="glass rounded-2xl shadow-large border border-white/30 p-8">
        <div className="text-center mb-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-2">📋 Choose Upload Method</h2>
          <p className="text-gray-600">Select how you'd like to import your FPL team</p>
        </div>
        
        <div className="flex justify-center space-x-6 mb-10">
          <button
            onClick={() => setUploadMethod('formation')}
            className={`px-8 py-4 rounded-2xl font-semibold transition-all duration-300 ${
              uploadMethod === 'formation'
                ? 'gradient-primary text-white shadow-large transform scale-105'
                : 'bg-white/80 text-gray-700 hover:bg-white hover:shadow-medium card-hover'
            }`}
          >
            <div className="flex flex-col items-center space-y-2">
              <span className="text-2xl">⚽</span>
              <span>Formation Entry</span>
              <span className="text-xs opacity-80">Click slots to select players</span>
            </div>
          </button>
          
          <button
            onClick={() => setUploadMethod('manager')}
            className={`px-8 py-4 rounded-2xl font-semibold transition-all duration-300 ${
              uploadMethod === 'manager'
                ? 'gradient-primary text-white shadow-large transform scale-105'
                : 'bg-white/80 text-gray-700 hover:bg-white hover:shadow-medium card-hover'
            }`}
          >
            <div className="flex flex-col items-center space-y-2">
              <span className="text-2xl">👤</span>
              <span>Manager Name</span>
              <span className="text-xs opacity-80">Search by your FPL name</span>
            </div>
          </button>
        </div>

        <div className="bg-white/50 rounded-xl p-6 backdrop-blur-sm">
          {uploadMethod === 'formation' ? (
            <FormationEntry 
              onTeamUpload={onTeamUpload} 
              loading={loading}
            />
          ) : (
            <ManagerEntry 
              onTeamUpload={onTeamUpload} 
              loading={loading}
            />
          )}
        </div>
      </div>
    </div>
  );
}

function FormationEntry({ onTeamUpload, loading }: {
  onTeamUpload: (data: { players: Player[], config: any }) => void;
  loading: boolean;
}) {
  const [selectedPlayers, setSelectedPlayers] = useState<Player[]>([]);
  const [startingXI, setStartingXI] = useState<number[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [teamConfig, setTeamConfig] = useState({
    free_transfers: 1,
    bank: 0.0,
    chips_available: ['Wildcard', 'Bench Boost', 'Triple Captain', 'Free Hit']
  });

  const handlePlayerChange = (player: Player | null, position: string, index: number) => {
    setSelectedPlayers(prev => {
      const newPlayers = [...prev];
      
      // Map frontend position codes to database position names for comparison
      const positionMap: { [key: string]: string } = {
        'GKP': 'Goalkeeper',
        'DEF': 'Defender',
        'MID': 'Midfielder',
        'FWD': 'Forward',
        'Goalkeeper': 'Goalkeeper',
        'Defender': 'Defender',
        'Midfielder': 'Midfielder',
        'Forward': 'Forward'
      };
      
      const normalizedPosition = positionMap[position] || position;
      
      // Find existing player in this specific slot
      const positionPlayers = prev.filter(p => {
        const playerPosition = positionMap[p.position] || p.position;
        return playerPosition === normalizedPosition;
      });
      
      // Remove the player currently in this slot
      if (positionPlayers[index]) {
        const playerToRemove = positionPlayers[index];
        const removeIndex = newPlayers.findIndex(p => p.id === playerToRemove.id);
        if (removeIndex !== -1) {
          newPlayers.splice(removeIndex, 1);
          console.log(`🔄 Replaced: ${playerToRemove.web_name} (${playerToRemove.position})`);
        }
        setStartingXI(prev => prev.filter(id => id !== playerToRemove.id));
      }
      
      // Add new player if provided
      if (player) {
        // Remove player if already selected elsewhere (prevent duplicates)
        const duplicateIndex = newPlayers.findIndex(p => p.id === player.id);
        if (duplicateIndex !== -1) {
          newPlayers.splice(duplicateIndex, 1);
          console.log(`⚠️ Removed duplicate: ${player.web_name}`);
          setStartingXI(prev => prev.filter(id => id !== player.id));
        }
        newPlayers.push(player);
        console.log(`✅ Added: ${player.web_name} (${player.position})`);
        setStartingXI(prev => {
          if (prev.includes(player.id)) {
            return prev;
          }
          if (prev.length < 11) {
            return [...prev, player.id];
          }
          return prev;
        });
      }
      
      return newPlayers;
    });
  };

  const handleAnalyze = () => {
    if (selectedPlayers.length !== 15) {
      setError(`Need exactly 15 players. You have ${selectedPlayers.length} selected.`);
      return;
    }

    // Validate team structure
    const validation = validateTeamStructure(selectedPlayers);
    if (!validation.isValid) {
      setError(validation.errors.join(', '));
      return;
    }

    setError(null);
    onTeamUpload({ 
      players: selectedPlayers, 
      config: {
        ...teamConfig,
        starting_xi: startingXI
      } 
    });
  };

  const toggleStartingXI = (playerId: number) => {
    setStartingXI(prev => {
      if (prev.includes(playerId)) {
        return prev.filter(id => id !== playerId);
      } else if (prev.length < 11) {
        return [...prev, playerId];
      } else {
        return prev;
      }
    });
  };

  const handleClear = () => {
    setSelectedPlayers([]);
    setError(null);
    setStartingXI([]);
  };

  return (
    <div className="space-y-6">
      {/* Team Configuration Panel */}
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl p-6 border-2 border-blue-200">
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
          ⚙️ Team Configuration
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Free Transfers */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Free Transfers
            </label>
            <input
              type="number"
              min="0"
              max="15"
              value={teamConfig.free_transfers}
              onChange={(e) => setTeamConfig(prev => ({
                ...prev,
                free_transfers: parseInt(e.target.value) || 0
              }))}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>
          
          {/* Money in Bank */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              💰 Money in Bank (£m)
            </label>
            <input
              type="number"
              step="0.1"
              min="0"
              max="100"
              value={teamConfig.bank}
              onChange={(e) => setTeamConfig(prev => ({
                ...prev,
                bank: parseFloat(e.target.value) || 0
              }))}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              placeholder="0.0"
            />
            <p className="mt-1 text-xs text-gray-500">
              Remaining budget after buying team
            </p>
          </div>
          
          {/* Starting XI Count */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Starting XI
            </label>
            <div className="px-3 py-2 bg-white border border-gray-300 rounded-lg text-center">
              <span className="text-2xl font-bold text-blue-600">{startingXI.length}</span>
              <span className="text-gray-500">/11</span>
            </div>
          </div>
        </div>
        
        {/* Chips Available */}
        <div className="mt-4">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Available Chips
          </label>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
            {['Wildcard', 'Bench Boost', 'Triple Captain', 'Free Hit'].map((chip) => (
              <label key={chip} className="flex items-center space-x-2 cursor-pointer bg-white p-2 rounded-lg border border-gray-200 hover:border-blue-300 transition-colors">
                <input
                  type="checkbox"
                  checked={teamConfig.chips_available.includes(chip)}
                  onChange={(e) => {
                    if (e.target.checked) {
                      setTeamConfig(prev => ({
                        ...prev,
                        chips_available: [...prev.chips_available, chip]
                      }));
                    } else {
                      setTeamConfig(prev => ({
                        ...prev,
                        chips_available: prev.chips_available.filter(c => c !== chip)
                      }));
                    }
                  }}
                  className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                />
                <span className="text-xs text-gray-700">{chip}</span>
              </label>
            ))}
          </div>
        </div>
        
        <div className="mt-4 space-y-2">
          <div className="p-3 bg-blue-100 rounded-lg border border-blue-200">
            <p className="text-xs text-blue-800">
              💡 <strong>Tip:</strong> Right-click on players to toggle Starting XI / Bench
            </p>
          </div>
          <div className="p-3 bg-green-100 rounded-lg border border-green-200">
            <p className="text-xs text-green-800">
              💰 <strong>Budget:</strong> Enter money left in bank. System calculates: Available = Bank + Player Out cost
            </p>
          </div>
        </div>
      </div>

      <div>
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          ⚽ Build Your Team (FPL Formation)
        </h3>
        <p className="text-gray-600 mb-6">
          Click slots to select players • Right-click to toggle Starting XI / Bench
        </p>
        
        <FormationLayout
          players={selectedPlayers}
          onPlayerChange={handlePlayerChange}
          startingXI={startingXI}
          onToggleStartingXI={toggleStartingXI}
        />
      </div>

      {/* Team Status */}
      <div className="flex items-center justify-between bg-gray-50 rounded-lg p-4">
        <div className="text-sm text-gray-600">
          Selected: <span className="font-medium">{selectedPlayers.length}/15 players</span>
          {selectedPlayers.length > 0 && (
            <span className="ml-4">
              Value: <span className="font-medium">
                £{(selectedPlayers.reduce((sum, p) => sum + p.now_cost, 0) / 10).toFixed(1)}M
              </span>
            </span>
          )}
        </div>
        
        <div className="flex space-x-3">
          <button
            onClick={handleClear}
            className="px-4 py-2 text-red-600 hover:text-red-700 border border-red-300 rounded-lg hover:bg-red-50 transition-colors"
          >
            🗑️ Clear
          </button>
          
          <button
            onClick={handleAnalyze}
            disabled={selectedPlayers.length !== 15 || loading}
            className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors font-medium"
          >
            {loading ? (
              <div className="flex items-center">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                Analyzing...
              </div>
            ) : (
              '🤖 Analyze Team'
            )}
          </button>
        </div>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-red-800 text-sm">{error}</p>
        </div>
      )}
    </div>
  );
}

function ManagerEntry({ onTeamUpload, loading }: {
  onTeamUpload: (data: { players: Player[], config: any }) => void;
  loading: boolean;
}) {
  const [managerName, setManagerName] = useState('');
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async () => {
    if (!managerName.trim()) {
      setError('Please enter your manager name');
      return;
    }

    if (managerName.length < 3) {
      setError('Manager name must be at least 3 characters');
      return;
    }

    setSearching(true);
    setError(null);
    setSearchResults([]);
    
    try {
      const response = await fetch(
        `http://localhost:8000/api/managers/search?name=${encodeURIComponent(managerName)}`
      );
      
      if (response.ok) {
        const data = await response.json();
        if (data.managers && data.managers.length > 0) {
          setSearchResults(data.managers);
        } else {
          setError('FPL doesn\'t support manager name search. Please use your Team ID instead or try Formation Entry.');
        }
      } else {
        setError('Search failed. Please try again.');
      }
    } catch (err) {
      setError('Connection error. Please check if backend is running.');
      console.error('Manager search error:', err);
    } finally {
      setSearching(false);
    }
  };

  const handleSelectManager = async (manager: any) => {
    setError(null);
    
    try {
      const response = await fetch(
        `http://localhost:8000/api/team/manager/${manager.id}`
      );
      
      if (response.ok) {
        const teamData = await response.json();
        if (teamData.players && teamData.players.length === 15) {
          onTeamUpload({ 
            players: teamData.players, 
            config: {
              free_transfers: 1,
              bank: 0.0,
              chips_available: ['Wildcard', 'Bench Boost', 'Triple Captain', 'Free Hit'],
              starting_xi: []
            }
          });
        } else {
          setError('Could not fetch complete team data. Please try Formation Entry.');
        }
      } else {
        const errorData = await response.json();
        setError(errorData.detail || 'Failed to fetch team data.');
      }
    } catch (err) {
      setError('Failed to fetch team. Please try again.');
      console.error('Team fetch error:', err);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          👤 Search by Manager Name
        </h3>
        
        <div className="space-y-4">
          <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                FPL Manager or Team Name:
              </label>
            <div className="flex space-x-2">
              <input
                type="text"
                className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
                placeholder="Enter your manager name or team name (e.g., 'nobody laxus')"
                value={managerName}
                onChange={(e) => setManagerName(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
              />
              <button
                onClick={handleSearch}
                disabled={searching || !managerName.trim()}
                className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 transition-colors"
              >
                {searching ? '🔍' : '🔍 Search'}
              </button>
            </div>
            <p className="mt-2 text-sm text-gray-500">
              Enter your manager name OR team name exactly as it appears on FPL
            </p>
          </div>

          {/* Search Results */}
          {searchResults.length > 0 && (
            <div className="bg-gray-50 rounded-lg p-4">
              <h4 className="font-medium text-gray-900 mb-3">Found Managers:</h4>
              <div className="space-y-2">
                {searchResults.map((manager, index) => (
                  <div
                    key={index}
                    className="bg-white rounded-lg p-3 border border-gray-200 hover:border-blue-300 cursor-pointer transition-colors"
                    onClick={() => handleSelectManager(manager)}
                  >
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="font-medium text-gray-900">{manager.player_name}</div>
                        <div className="text-sm text-gray-500">{manager.team_name}</div>
                      </div>
                      <div className="text-right">
                        <div className="text-sm font-medium text-gray-900">
                          Rank: {manager.overall_rank?.toLocaleString() || 'Unknown'}
                        </div>
                        <div className="text-xs text-gray-500">
                          {manager.total_points || 0} points
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {error && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
          <p className="text-yellow-800 text-sm">{error}</p>
        </div>
      )}

      {/* Instructions */}
      <div className="bg-blue-50 rounded-lg p-4">
        <h4 className="font-medium text-blue-900 mb-2">💡 How to use manager search:</h4>
        <ul className="text-sm text-blue-800 space-y-1">
          <li>• Enter your manager name as it appears on FPL</li>
          <li>• Click search to find matching managers</li>
          <li>• Select your manager from the results</li>
          <li>• Your team will be automatically loaded</li>
        </ul>
        <div className="mt-3 p-2 bg-blue-100 rounded text-xs text-blue-700">
          💡 <strong>Note:</strong> Manager search uses FPL's internal systems. If you can't find yourself, try the Formation Entry method instead.
        </div>
      </div>
    </div>
  );
}
