'use client';

import { useState } from 'react';
import dynamic from 'next/dynamic';
import { Player, ManagerSummary, ManagerTeam } from '@/lib/types';
import { validateTeamStructure, formatPrice } from '@/lib/utils';
import { FPLApi } from '@/lib/api';
import { useNotifications } from '@/components/ui/NotificationProvider';
import { Skeleton } from '@/components/ui/Skeleton';

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

type UploadMethod = 'formation' | 'manager';

const DEFAULT_CHIPS = ['Wildcard', 'Bench Boost', 'Triple Captain', 'Free Hit'] as const;

interface TeamConfigState {
  free_transfers: number;
  bank: number;
  chips_available: string[];
}

interface TeamUploadProps {
  onTeamUpload: (data: { players: Player[], config: any }) => void;
  loading: boolean;
}

export default function TeamUpload({ onTeamUpload, loading }: TeamUploadProps) {
  const [uploadMethod, setUploadMethod] = useState<UploadMethod>('formation');

  return (
    <div className="space-y-6">
      <div className="space-y-3">
        <p className="text-sm font-semibold uppercase tracking-wide text-slate-500">Upload options</p>
        <div className="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <h2 className="text-2xl font-semibold text-slate-900">Choose how you want to load your squad</h2>
            <p className="text-sm text-slate-500">Switch between the interactive builder or manager lookup.</p>
          </div>
          <UploadMethodSelector value={uploadMethod} onChange={setUploadMethod} />
        </div>
        </div>

      <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm sm:p-6">
          {uploadMethod === 'formation' ? (
          <FormationEntry onTeamUpload={onTeamUpload} loading={loading} />
          ) : (
          <ManagerEntry onTeamUpload={onTeamUpload} loading={loading} />
          )}
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
  const [teamConfig, setTeamConfig] = useState<TeamConfigState>({
    free_transfers: 1,
    bank: 0.0,
    chips_available: [...DEFAULT_CHIPS]
  });
  const [viewMode, setViewMode] = useState<'pitch' | 'list'>('pitch');
  const { notify } = useNotifications();

  const selectedValue = selectedPlayers.reduce((sum, p) => sum + p.now_cost, 0) / 10;

  const handleConfigChange = (updates: Partial<TeamConfigState>) => {
    setTeamConfig((prev) => ({ ...prev, ...updates }));
  };

  const handleChipToggle = (chip: string, enabled: boolean) => {
    setTeamConfig((prev) => ({
      ...prev,
      chips_available: enabled
        ? [...prev.chips_available, chip]
        : prev.chips_available.filter((c) => c !== chip)
    }));
  };

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
        }
        setStartingXI(prev => prev.filter(id => id !== playerToRemove.id));
      }
      
      // Add new player if provided
      if (player) {
        // Remove player if already selected elsewhere (prevent duplicates)
        const duplicateIndex = newPlayers.findIndex(p => p.id === player.id);
        if (duplicateIndex !== -1) {
          newPlayers.splice(duplicateIndex, 1);
          setStartingXI(prev => prev.filter(id => id !== player.id));
        }
        newPlayers.push(player);
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
      const message = `Need exactly 15 players. You have ${selectedPlayers.length} selected.`;
      setError(message);
      notify(message, { variant: 'warning' });
      return;
    }

    // Validate team structure
    const validation = validateTeamStructure(selectedPlayers);
    if (!validation.isValid) {
      const message = validation.errors.join(', ');
      setError(message);
      notify(message, { variant: 'error' });
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
    notify('Team validated. Redirecting to analysis...', { variant: 'success' });
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
    notify('Selection cleared', { variant: 'info', duration: 2000 });
  };

  return (
    <div className="space-y-6">
      <TeamConfigPanel
        teamConfig={teamConfig}
        startingXICount={startingXI.length}
        onConfigChange={handleConfigChange}
        onChipToggle={handleChipToggle}
      />
        
      <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm sm:p-6">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h3 className="text-lg font-semibold text-slate-900">Build your squad</h3>
            <p className="text-sm text-slate-500">Click a slot to pick a player. Right-click or long press to move between XI and bench.</p>
          </div>
          <ViewModeToggle value={viewMode} onChange={setViewMode} />
        </div>
        
        <div className="mt-4">
          {viewMode === 'pitch' ? (
        <FormationLayout
          players={selectedPlayers}
          onPlayerChange={handlePlayerChange}
          startingXI={startingXI}
          onToggleStartingXI={toggleStartingXI}
        />
          ) : (
            <CompactRoster
              players={selectedPlayers}
              startingXI={startingXI}
              onToggleStartingXI={toggleStartingXI}
              onRemovePlayer={(player) => {
                setSelectedPlayers((prev) => prev.filter((p) => p.id !== player.id));
                setStartingXI((prev) => prev.filter((id) => id !== player.id));
              }}
            />
          )}
        </div>
      </div>

      <TeamStatusBar
        selectedCount={selectedPlayers.length}
        totalValue={selectedValue}
        onClear={handleClear}
        onAnalyze={handleAnalyze}
        disableAnalyze={selectedPlayers.length !== 15 || loading}
        loading={loading}
      />

      {error && (
        <div className="rounded-xl border border-red-200 bg-red-50/70 p-4 text-sm text-red-700">
          {error}
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
  const [searchResults, setSearchResults] = useState<ManagerSummary[]>([]);
  const [searching, setSearching] = useState(false);
  const [fetchingTeamId, setFetchingTeamId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [infoMessage, setInfoMessage] = useState<string | null>(null);

  const handleSearch = async () => {
    if (!managerName.trim()) {
      setError('Please enter your manager or team name');
      return;
    }

    if (managerName.length < 3) {
      setError('Manager name must be at least 3 characters');
      return;
    }

    setSearching(true);
    setError(null);
    setInfoMessage(null);
    setSearchResults([]);
    
    try {
      const response = await FPLApi.searchManager(managerName.trim());
      const managers = response?.managers || [];
      setSearchResults(managers);
      if (!managers.length) {
        setInfoMessage(response?.message || 'No managers found. Try a different spelling.');
      }
    } catch (err) {
      console.error('Manager search error:', err);
      setError('Search failed. Please confirm the backend is running.');
    } finally {
      setSearching(false);
    }
  };

  const deriveConfigFromManagerTeam = (teamData: ManagerTeam) => {
    const startingXIFromResponse =
      Array.isArray(teamData.starting_xi) && teamData.starting_xi.length
        ? teamData.starting_xi
        : teamData.players
            .filter((player) => typeof player.pick_position === 'number' && player.pick_position <= 11)
            .map((player) => player.id);
    const normalizedStartingXI = startingXIFromResponse.slice(0, 11);

    const chipsFromResponse =
      Array.isArray(teamData.chips_available) && teamData.chips_available.length
        ? [...teamData.chips_available]
        : [...DEFAULT_CHIPS];

    const bankValue = typeof teamData.bank === 'number' ? teamData.bank : 0;
    const freeTransfersValue =
      typeof teamData.free_transfers === 'number' && teamData.free_transfers > 0
        ? teamData.free_transfers
        : 1;

    return {
      free_transfers: freeTransfersValue,
      bank: Number(bankValue.toFixed(1)),
      chips_available: chipsFromResponse,
      starting_xi: normalizedStartingXI
    };
  };

  const handleSelectManager = async (manager: ManagerSummary) => {
    setError(null);
    setInfoMessage(null);
    const managerId = manager?.id?.toString();
    if (!managerId) {
      setError('Invalid manager selection.');
      return;
    }

    setFetchingTeamId(managerId);
    
    try {
      const teamData = await FPLApi.getTeamByManagerId(managerId);
      const players = teamData?.players || [];
      if (players.length === 15) {
        const hydratedConfig = deriveConfigFromManagerTeam(teamData);
          onTeamUpload({ 
          players, 
          config: hydratedConfig
        });
      } else {
        setError('Could not fetch a complete squad. Please try Formation Entry.');
      }
    } catch (err: any) {
      console.error('Team fetch error:', err);
      const backendMessage = err?.response?.data?.detail;
      setError(backendMessage || 'Failed to fetch team. Please try again.');
    } finally {
      setFetchingTeamId(null);
    }
  };

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <h3 className="text-lg font-semibold text-slate-900">Search by manager or team name</h3>
        <p className="text-sm text-slate-500">Enter your official FPL manager or team name and select the matching record.</p>
      </div>
        
        <div className="space-y-4">
          <div>
          <label className="text-sm font-medium text-slate-700">FPL manager or team name</label>
          <div className="mt-2 flex flex-col gap-3 sm:flex-row">
              <input
                type="text"
              className="flex-1 rounded-lg border border-slate-200 px-4 py-2.5 text-sm text-slate-900 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-100"
              placeholder="e.g. The Data XI"
                value={managerName}
                onChange={(e) => setManagerName(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
              />
              <button
                onClick={handleSearch}
              disabled={searching || !managerName.trim() || loading}
              className="btn-hover rounded-lg bg-slate-900 px-4 py-2.5 text-sm font-semibold text-white shadow-sm transition-colors hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-60"
              >
              {searching ? 'Searching...' : 'Search'}
              </button>
            </div>
          <p className="mt-2 text-xs text-slate-500">Match the spelling exactly as it appears on fantasy.premierleague.com.</p>
        </div>

        {infoMessage && (
          <div className="rounded-xl border border-blue-100 bg-blue-50/70 p-4 text-sm text-blue-800">{infoMessage}</div>
        )}

        {searching && (
          <div className="space-y-3 rounded-2xl border border-slate-200 bg-slate-50/70 p-4">
            {[1, 2, 3].map((item) => (
              <Skeleton key={item} className="h-14 w-full rounded-xl" />
            ))}
          </div>
        )}

        {!searching && searchResults.length > 0 && (
          <div className="space-y-2 rounded-2xl border border-slate-200 bg-white p-4">
            <h4 className="text-sm font-semibold text-slate-800">Found managers</h4>
              <div className="space-y-2">
              {searchResults.map((manager, index) => {
                const managerId = manager.id?.toString() || String(index);
                const isLoading = fetchingTeamId === managerId;
                return (
                  <button
                    key={managerId}
                    className="w-full rounded-xl border border-slate-200 px-4 py-3 text-left transition-colors hover:border-blue-200 disabled:cursor-not-allowed"
                    onClick={() => handleSelectManager(manager)}
                    disabled={!!fetchingTeamId}
                  >
                    <div className="flex items-center justify-between gap-3">
                      <div>
                        <p className="text-sm font-semibold text-slate-900">{manager.player_name}</p>
                        <p className="text-xs text-slate-500">{manager.team_name}</p>
                      </div>
                      <div className="text-right text-xs text-slate-500">
                        <p className="font-semibold text-slate-800">Rank {manager.overall_rank?.toLocaleString() ?? '--'}</p>
                        <p>{manager.total_points || 0} pts</p>
                        {isLoading && <p className="text-blue-500">Loading...</p>}
                      </div>
                    </div>
                  </button>
                );
              })}
              </div>
            </div>
          )}
      </div>

      {error && (
        <div className="rounded-xl border border-amber-200 bg-amber-50/80 p-4 text-sm text-amber-800">
          {error}
        </div>
      )}

      <div className="rounded-2xl border border-blue-100 bg-blue-50/70 p-4 text-sm text-blue-800">
        <p className="font-semibold text-blue-900">Tips</p>
        <ul className="mt-2 list-disc space-y-1 pl-4">
          <li>Search using the exact manager or team name.</li>
          <li>Select the row matching your rank/points to auto-load the squad.</li>
          <li>If no results appear, try the formation builder instead.</li>
        </ul>
      </div>
    </div>
  );
}

interface UploadMethodSelectorProps {
  value: UploadMethod;
  onChange: (method: UploadMethod) => void;
}

function UploadMethodSelector({ value, onChange }: UploadMethodSelectorProps) {
  const options: Array<{ key: UploadMethod; title: string; subtitle: string }> = [
    { key: 'formation', title: 'Formation builder', subtitle: 'Select players manually on the pitch' },
    { key: 'manager', title: 'Manager lookup', subtitle: 'Search by your FPL name or team' }
  ];

  return (
    <div className="inline-flex rounded-full border border-slate-200 bg-slate-50 p-1 text-sm font-medium text-slate-600">
      {options.map((option) => (
        <button
          key={option.key}
          onClick={() => onChange(option.key)}
          className={`flex flex-1 flex-col rounded-full px-5 py-2 text-left transition-colors ${
            value === option.key ? 'bg-white text-slate-900 shadow-sm' : 'hover:text-slate-900'
          }`}
        >
          <span>{option.title}</span>
          <span className="text-xs font-normal text-slate-500">{option.subtitle}</span>
        </button>
      ))}
    </div>
  );
}

interface TeamConfigPanelProps {
  teamConfig: TeamConfigState;
  startingXICount: number;
  onConfigChange: (updates: Partial<TeamConfigState>) => void;
  onChipToggle: (chip: string, enabled: boolean) => void;
}

function TeamConfigPanel({ teamConfig, startingXICount, onConfigChange, onChipToggle }: TeamConfigPanelProps) {
  const chips = ['Wildcard', 'Bench Boost', 'Triple Captain', 'Free Hit'];
  return (
    <div className="rounded-2xl border border-slate-200 bg-slate-50/60 p-4 sm:p-6">
      <div className="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
        <h3 className="text-base font-semibold text-slate-900">Team configuration</h3>
        <p className="text-sm text-slate-500">Update the squad assumptions before running analysis.</p>
      </div>

      <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-3">
        <ConfigField label="Free transfers">
          <input
            type="number"
            min="0"
            max="15"
            value={teamConfig.free_transfers}
            onChange={(e) => onConfigChange({ free_transfers: Math.max(0, parseInt(e.target.value, 10) || 0) })}
            className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-900 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-100"
          />
        </ConfigField>

        <ConfigField label="Money in bank (GBP m)" hint="Remaining budget after trades">
          <input
            type="number"
            step="0.1"
            min="0"
            max="100"
            value={teamConfig.bank}
            onChange={(e) => onConfigChange({ bank: Math.max(0, parseFloat(e.target.value) || 0) })}
            className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-900 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-100"
            placeholder="0.0"
          />
        </ConfigField>

        <ConfigField label="Starting XI selected" hint="Tap a player to move them to the bench">
          <div className="flex items-center justify-between rounded-lg border border-white/70 bg-white px-4 py-2 text-sm font-semibold text-slate-800">
            <span>{startingXICount} of 11</span>
            <span className="text-xs text-slate-500">Valid formation required</span>
          </div>
        </ConfigField>
      </div>

      <div className="mt-4">
        <label className="text-sm font-medium text-slate-700">Available chips</label>
        <div className="mt-2 grid grid-cols-2 gap-2 sm:grid-cols-4">
          {chips.map((chip) => {
            const isChecked = teamConfig.chips_available.includes(chip);
            return (
              <button
                key={chip}
                type="button"
                onClick={() => onChipToggle(chip, !isChecked)}
                className={`rounded-xl border px-3 py-2 text-xs font-semibold transition-colors ${
                  isChecked ? 'border-blue-200 bg-white text-blue-700' : 'border-slate-200 bg-white/70 text-slate-600'
                }`}
              >
                {chip}
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}

interface TeamStatusBarProps {
  selectedCount: number;
  totalValue: number;
  onClear: () => void;
  onAnalyze: () => void;
  disableAnalyze: boolean;
  loading: boolean;
}

interface ViewModeToggleProps {
  value: 'pitch' | 'list';
  onChange: (mode: 'pitch' | 'list') => void;
}

function ViewModeToggle({ value, onChange }: ViewModeToggleProps) {
  return (
    <div className="flex rounded-full border border-slate-200 bg-slate-50 p-1 text-sm font-medium text-slate-600">
      {['pitch', 'list'].map((mode) => (
        <button
          key={mode}
          type="button"
          onClick={() => onChange(mode as 'pitch' | 'list')}
          className={`flex-1 rounded-full px-3 py-1 capitalize transition-colors ${
            value === mode ? 'bg-white text-slate-900 shadow-sm' : 'hover:text-slate-900'
          }`}
        >
          {mode === 'pitch' ? 'Pitch view' : 'List view'}
        </button>
      ))}
    </div>
  );
}

interface CompactRosterProps {
  players: Player[];
  startingXI: number[];
  onToggleStartingXI: (playerId: number) => void;
  onRemovePlayer: (player: Player) => void;
}

function CompactRoster({ players, startingXI, onToggleStartingXI, onRemovePlayer }: CompactRosterProps) {
  if (players.length === 0) {
    return <p className="rounded-xl border border-dashed border-slate-200 p-4 text-center text-sm text-slate-500">No players selected yet.</p>;
  }

  const order = ['Goalkeeper', 'GKP', 'Defender', 'DEF', 'Midfielder', 'MID', 'Forward', 'FWD'];
  const sorted = [...players].sort((a, b) => order.indexOf(a.position) - order.indexOf(b.position));

  return (
    <div className="mt-4">
      <div className="hidden overflow-hidden rounded-xl border border-slate-200 md:block">
        <table className="min-w-full divide-y divide-slate-200 text-sm">
          <thead className="bg-slate-50 text-left text-xs font-semibold uppercase tracking-wide text-slate-500">
            <tr>
              <th className="px-4 py-3">Player</th>
              <th className="px-4 py-3">Team</th>
              <th className="px-4 py-3">Position</th>
              <th className="px-4 py-3">Price</th>
              <th className="px-4 py-3">Status</th>
              <th className="px-4 py-3 text-right">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 bg-white text-slate-700">
            {sorted.map((player) => {
              const inXI = startingXI.includes(player.id);
              return (
                <tr key={player.id}>
                  <td className="px-4 py-3 font-medium">{player.web_name}</td>
                  <td className="px-4 py-3 text-slate-500">{player.team_name}</td>
                  <td className="px-4 py-3 text-slate-500">{player.position}</td>
                  <td className="px-4 py-3">{formatPrice(player.now_cost)}</td>
                  <td className="px-4 py-3">
                    <span className={`rounded-full px-2 py-1 text-xs font-semibold ${inXI ? 'bg-green-50 text-green-700' : 'bg-slate-100 text-slate-600'}`}>
                      {inXI ? 'Starting' : 'Bench'}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-right">
                    <div className="flex justify-end gap-2">
                      <button
                        type="button"
                        onClick={() => onToggleStartingXI(player.id)}
                        className="text-xs font-semibold text-blue-600 hover:text-blue-800"
                      >
                        {inXI ? 'Move to bench' : 'Add to XI'}
                      </button>
                      <button
                        type="button"
                        onClick={() => onRemovePlayer(player)}
                        className="text-xs font-semibold text-red-500 hover:text-red-700"
                      >
                        Remove
                      </button>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="space-y-3 md:hidden">
        {sorted.map((player) => {
          const inXI = startingXI.includes(player.id);
          return (
            <div key={player.id} className="rounded-xl border border-slate-200 bg-white p-3">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-semibold text-slate-900">{player.web_name}</p>
                  <p className="text-xs text-slate-500">{player.team_name}</p>
                </div>
                <span className={`rounded-full px-2 py-1 text-xs font-semibold ${inXI ? 'bg-green-50 text-green-700' : 'bg-slate-100 text-slate-600'}`}>
                  {inXI ? 'Starting' : 'Bench'}
                </span>
              </div>
              <div className="mt-2 flex items-center justify-between text-xs text-slate-500">
                <span>{player.position}</span>
                <span>{formatPrice(player.now_cost)}</span>
              </div>
              <div className="mt-3 flex gap-2 text-xs font-semibold">
                <button
                  type="button"
                  onClick={() => onToggleStartingXI(player.id)}
                  className="flex-1 rounded-lg border border-slate-200 px-2 py-1 text-center text-blue-600"
                >
                  {inXI ? 'Move to bench' : 'Add to XI'}
                </button>
                <button
                  type="button"
                  onClick={() => onRemovePlayer(player)}
                  className="flex-1 rounded-lg border border-slate-200 px-2 py-1 text-center text-red-500"
                >
                  Remove
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function TeamStatusBar({ selectedCount, totalValue, onClear, onAnalyze, disableAnalyze, loading }: TeamStatusBarProps) {
  return (
    <div className="flex flex-col gap-4 rounded-2xl border border-slate-200 bg-white p-4 shadow-sm sm:flex-row sm:items-center sm:justify-between">
      <div>
        <p className="text-sm text-slate-500">Selection summary</p>
        <div className="mt-1 flex flex-wrap items-center gap-4 text-sm font-semibold text-slate-900">
          <span>{selectedCount}/15 players selected</span>
          {selectedCount > 0 && <span className="text-slate-500">•</span>}
          {selectedCount > 0 && <span>Team value {totalValue.toFixed(1)}m</span>}
        </div>
      </div>

      <div className="flex flex-wrap gap-3">
        <button
          onClick={onClear}
          className="rounded-lg border border-slate-200 px-4 py-2 text-sm font-medium text-slate-700 transition-colors hover:bg-slate-50"
        >
          Reset selection
        </button>

        <button
          onClick={onAnalyze}
          disabled={disableAnalyze}
          className="btn-hover rounded-lg bg-blue-600 px-5 py-2 text-sm font-semibold text-white shadow-sm transition-colors hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {loading ? (
            <span className="flex items-center gap-2">
              <span className="loading-spinner h-4 w-4 rounded-full border border-white/60 border-t-transparent"></span>
              Analyzing
            </span>
          ) : (
            'Run analysis'
          )}
        </button>
      </div>
    </div>
  );
}

function ConfigField({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div className="space-y-2 rounded-xl border border-white/60 bg-white px-3 py-3">
      <div className="space-y-1">
        <p className="text-sm font-medium text-slate-800">{label}</p>
        {hint && <p className="text-xs text-slate-500">{hint}</p>}
      </div>
      {children}
    </div>
  );
}
