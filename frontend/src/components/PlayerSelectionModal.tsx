'use client';

import { useState, useEffect, Fragment, useCallback, useRef } from 'react';
import { Dialog, Transition, Listbox } from '@headlessui/react';
import { XMarkIcon, ChevronUpDownIcon, CheckIcon, MagnifyingGlassIcon } from '@heroicons/react/24/outline';
import { Player, Club } from '@/lib/types';
import { formatPrice, getPositionIcon, getPlayerStatus, classNames } from '@/lib/utils';
import { FPLApi } from '@/lib/api';

interface PlayerSelectionModalProps {
  isOpen: boolean;
  onClose: () => void;
  position: string;
  onPlayerSelect: (player: Player) => void;
  currentPlayer?: Player | null;
  batchMode?: boolean;
  selectedPlayers?: Player[];
}

const CACHE_TTL_MS = 5 * 60 * 1000;

export default function PlayerSelectionModal({ 
  isOpen, 
  onClose, 
  position, 
  onPlayerSelect, 
  currentPlayer,
  batchMode = false,
  selectedPlayers = []
}: PlayerSelectionModalProps) {
  const [players, setPlayers] = useState<Player[]>([]);
  const [filteredPlayers, setFilteredPlayers] = useState<Player[]>([]);
  const [teams, setTeams] = useState<Club[]>([]);
  const [loading, setLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedTeam, setSelectedTeam] = useState<Club | null>(null);
  const [sortBy, setSortBy] = useState<'points' | 'price' | 'form' | 'name'>('points');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  const [multiSelectMode, setMultiSelectMode] = useState(batchMode);
  const [tempSelectedPlayers, setTempSelectedPlayers] = useState<Player[]>([]);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [minPrice, setMinPrice] = useState('');
  const [maxPrice, setMaxPrice] = useState('');
  const [onlyAvailable, setOnlyAvailable] = useState(false);

  const playerCacheRef = useRef<Map<string, { data: Player[]; timestamp: number }>>(new Map());
  const teamsCacheRef = useRef<{ data: Club[]; timestamp: number }>({ data: [], timestamp: 0 });

  const fetchData = useCallback(async () => {
    const positionMap = {
      GKP: 'Goalkeeper',
      DEF: 'Defender',
      MID: 'Midfielder',
      FWD: 'Forward'
    };
    
    const dbPosition = positionMap[position as keyof typeof positionMap] || position;
    setLoading(true);
    setFetchError(null);
    
    try {
      const now = Date.now();
      const cachedPlayers = playerCacheRef.current.get(dbPosition);
      const playerCacheFresh = cachedPlayers && now - cachedPlayers.timestamp < CACHE_TTL_MS;
      const teamCacheFresh = teamsCacheRef.current.data.length > 0 && now - teamsCacheRef.current.timestamp < CACHE_TTL_MS;
      
      let playersData: Player[] = [];
      if (playerCacheFresh && cachedPlayers) {
        playersData = cachedPlayers.data;
      } else {
        playersData = await FPLApi.getPlayers({ position: dbPosition });
        playerCacheRef.current.set(dbPosition, { data: playersData, timestamp: now });
      }
      
      let teamsData: Club[] = [];
      if (teamCacheFresh) {
        teamsData = teamsCacheRef.current.data;
      } else {
        teamsData = await FPLApi.getTeams();
        teamsCacheRef.current = { data: teamsData, timestamp: now };
      }
      
      setPlayers(playersData || []);
      setTeams(teamsData || []);
    } catch (error) {
      console.error('Failed to load player data', error);
      setFetchError('Unable to load player data. Please verify the backend is running and try again.');
      setPlayers([]);
      setFilteredPlayers([]);
      setTeams([]);
    } finally {
      setLoading(false);
    }
  }, [position]);

  // Fetch players and teams when modal opens
  useEffect(() => {
    if (isOpen) {
      fetchData();
    }
  }, [isOpen, position, fetchData]);

  // Reset state when modal closes
  useEffect(() => {
    if (!isOpen) {
      setPlayers([]);
      setFilteredPlayers([]);
      setTeams([]);
      setSearchQuery('');
      setSelectedTeam(null);
      setFetchError(null);
    }
  }, [isOpen]);

  // Filter and sort players
  useEffect(() => {
    let filtered = [...players];

    // Filter by search query
    if (searchQuery) {
      filtered = filtered.filter(player =>
        player.web_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        player.team_name.toLowerCase().includes(searchQuery.toLowerCase())
      );
    }

    // Filter by team
    if (selectedTeam) {
      filtered = filtered.filter(player => player.team === selectedTeam.id);
    }

    if (minPrice) {
      const min = parseFloat(minPrice);
      if (!Number.isNaN(min)) {
        filtered = filtered.filter(player => player.now_cost / 10 >= min);
      }
    }

    if (maxPrice) {
      const max = parseFloat(maxPrice);
      if (!Number.isNaN(max)) {
        filtered = filtered.filter(player => player.now_cost / 10 <= max);
      }
    }

    if (onlyAvailable) {
      filtered = filtered.filter(player => player.status === 'a');
    }

    // Sort players
    filtered.sort((a, b) => {
      let aValue, bValue;
      
      switch (sortBy) {
        case 'points':
          aValue = a.total_points;
          bValue = b.total_points;
          break;
        case 'price':
          aValue = a.now_cost;
          bValue = b.now_cost;
          break;
        case 'form':
          aValue = parseFloat(a.form?.toString() || '0');
          bValue = parseFloat(b.form?.toString() || '0');
          break;
        case 'name':
          aValue = a.web_name.toLowerCase();
          bValue = b.web_name.toLowerCase();
          break;
        default:
          aValue = a.total_points;
          bValue = b.total_points;
      }

      if (sortOrder === 'desc') {
        return bValue > aValue ? 1 : -1;
      } else {
        return aValue > bValue ? 1 : -1;
      }
    });

    setFilteredPlayers(filtered);
  }, [players, searchQuery, selectedTeam, sortBy, sortOrder, minPrice, maxPrice, onlyAvailable]);

  const handlePlayerSelect = (player: Player) => {
    if (multiSelectMode) {
      // In multi-select mode, add/remove from temp selection
      const isAlreadySelected = tempSelectedPlayers.some(p => p.id === player.id);
      if (isAlreadySelected) {
        setTempSelectedPlayers(tempSelectedPlayers.filter(p => p.id !== player.id));
      } else {
        setTempSelectedPlayers([...tempSelectedPlayers, player]);
      }
    } else {
      // Single select mode - immediate selection and close
      onPlayerSelect(player);
      onClose();
    }
  };

  const handleConfirmBatch = () => {
    // Send all selected players one by one with a small delay to ensure state updates
    if (tempSelectedPlayers.length === 0) {
      return;
    }

    tempSelectedPlayers.forEach((player) => {
      onPlayerSelect(player);
    });
    setTempSelectedPlayers([]);
    onClose();
  };

  const handleCancelBatch = () => {
    setTempSelectedPlayers([]);
    onClose();
  };

  const getPositionColor = (pos: string) => {
    switch (pos) {
      case 'GKP': return 'text-green-600 bg-green-50';
      case 'DEF': return 'text-yellow-600 bg-yellow-50';
      case 'MID': return 'text-blue-600 bg-blue-50';
      case 'FWD': return 'text-red-600 bg-red-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  const sortOptions = [
    { value: 'points', label: 'Total Points' },
    { value: 'price', label: 'Price' },
    { value: 'form', label: 'Form' },
    { value: 'name', label: 'Name' }
  ];

  return (
    <Transition appear show={isOpen} as={Fragment}>
      <Dialog as="div" className="relative z-50" onClose={onClose}>
        <Transition.Child
          as={Fragment}
          enter="ease-out duration-300"
          enterFrom="opacity-0"
          enterTo="opacity-100"
          leave="ease-in duration-200"
          leaveFrom="opacity-100"
          leaveTo="opacity-0"
        >
          <div className="fixed inset-0 bg-black/50 backdrop-blur-sm" />
        </Transition.Child>

        <div className="fixed inset-0 overflow-y-auto">
          <div className="flex min-h-full items-center justify-center p-4 text-center">
            <Transition.Child
              as={Fragment}
              enter="ease-out duration-300"
              enterFrom="opacity-0 scale-95"
              enterTo="opacity-100 scale-100"
              leave="ease-in duration-200"
              leaveFrom="opacity-100 scale-100"
              leaveTo="opacity-0 scale-95"
            >
              <Dialog.Panel className="w-full max-w-4xl transform overflow-hidden rounded-3xl bg-white shadow-2xl transition-all">
                
                {/* Header */}
                <div className="bg-gradient-to-r from-blue-600 to-purple-600 px-8 py-6">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-4">
                      <div className={`w-12 h-12 rounded-full flex items-center justify-center text-2xl ${getPositionColor(position)}`}>
                        {getPositionIcon(position)}
                      </div>
                      <div className="text-left">
                        <Dialog.Title className="text-2xl font-bold text-white">
                          Select {position === 'GKP' ? 'Goalkeeper' : 
                                 position === 'DEF' ? 'Defender' :
                                 position === 'MID' ? 'Midfielder' : 'Forward'}
                        </Dialog.Title>
                        <p className="text-blue-100 text-sm">
                          {multiSelectMode 
                            ? `Batch Selection: ${tempSelectedPlayers.length} players selected`
                            : 'Choose a player for your team'}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-3">
                      {/* Batch Mode Toggle */}
                      <button
                        onClick={() => {
                          setMultiSelectMode(!multiSelectMode);
                          setTempSelectedPlayers([]);
                        }}
                        className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                          multiSelectMode 
                            ? 'bg-white text-blue-600' 
                            : 'bg-white/20 text-white hover:bg-white/30'
                        }`}
                      >
                        {multiSelectMode ? '✓ Batch Mode' : 'Batch Mode'}
                      </button>
                      <button
                        onClick={multiSelectMode ? handleCancelBatch : onClose}
                        className="rounded-full bg-white/20 p-2 text-white hover:bg-white/30 transition-colors"
                      >
                        <XMarkIcon className="h-6 w-6" />
                      </button>
                    </div>
                  </div>
                </div>

                {/* Filters */}
                <div className="border-b bg-slate-50 px-6 py-4">
                  <div className="flex flex-wrap items-center gap-4">
                    <div className="min-w-60 flex-1">
                      <div className="relative">
                        <MagnifyingGlassIcon className="pointer-events-none absolute left-3 top-1/2 h-5 w-5 -translate-y-1/2 text-slate-400" />
                        <input
                          type="text"
                          placeholder="Search players..."
                          value={searchQuery}
                          onChange={(e) => setSearchQuery(e.target.value)}
                          className="w-full rounded-xl border border-slate-200 bg-white pl-10 pr-4 py-2.5 text-sm text-slate-900 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-100"
                        />
                      </div>
                    </div>

                    <div className="min-w-48">
                      <Listbox value={selectedTeam} onChange={setSelectedTeam}>
                        <div className="relative">
                          <Listbox.Button className="relative w-full cursor-pointer rounded-xl border border-slate-200 bg-white py-2.5 pl-4 pr-10 text-left text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-blue-100">
                            <span className="block truncate">
                              {selectedTeam ? selectedTeam.name : 'All teams'}
                            </span>
                            <span className="pointer-events-none absolute inset-y-0 right-0 flex items-center pr-3">
                              <ChevronUpDownIcon className="h-5 w-5 text-slate-400" />
                            </span>
                          </Listbox.Button>
                          <Transition
                            as={Fragment}
                            leave="transition ease-in duration-100"
                            leaveFrom="opacity-100"
                            leaveTo="opacity-0"
                          >
                            <Listbox.Options className="absolute z-20 mt-1 max-h-60 w-full overflow-auto rounded-xl border border-slate-100 bg-white py-1 text-sm shadow-xl focus:outline-none">
                              <Listbox.Option
                                value={null}
                                className={({ active }) =>
                                  classNames(
                                    active ? 'bg-blue-50 text-blue-900' : 'text-slate-900',
                                    'cursor-pointer select-none px-4 py-2'
                                  )
                                }
                              >
                                All teams
                              </Listbox.Option>
                              {teams.map((team) => (
                                <Listbox.Option
                                  key={team.id}
                                  value={team}
                                  className={({ active }) =>
                                    classNames(
                                      active ? 'bg-blue-50 text-blue-900' : 'text-slate-900',
                                      'cursor-pointer select-none px-4 py-2'
                                    )
                                  }
                                >
                                  {({ selected }) => (
                                    <div className="flex items-center justify-between">
                                      <span className={selected ? 'font-semibold' : 'font-normal'}>
                                        {team.name}
                                      </span>
                                      {selected && <CheckIcon className="h-4 w-4 text-blue-600" />}
                                    </div>
                                  )}
                                </Listbox.Option>
                              ))}
                            </Listbox.Options>
                          </Transition>
                        </div>
                      </Listbox>
                    </div>

                    <div className="flex items-center gap-2">
                      <Listbox value={sortBy} onChange={setSortBy}>
                        <div className="relative">
                          <Listbox.Button className="min-w-40 rounded-xl border border-slate-200 bg-white py-2.5 pl-4 pr-10 text-left text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-blue-100">
                            <span className="block truncate">
                              Sort by {sortOptions.find(opt => opt.value === sortBy)?.label}
                            </span>
                            <span className="pointer-events-none absolute inset-y-0 right-0 flex items-center pr-3">
                              <ChevronUpDownIcon className="h-5 w-5 text-slate-400" />
                            </span>
                          </Listbox.Button>
                          <Listbox.Options className="absolute z-20 mt-1 max-h-60 w-full overflow-auto rounded-xl border border-slate-100 bg-white py-1 text-sm shadow-xl focus:outline-none">
                            {sortOptions.map((option) => (
                              <Listbox.Option
                                key={option.value}
                                value={option.value}
                                className={({ active }) =>
                                  classNames(
                                    active ? 'bg-blue-50 text-blue-900' : 'text-slate-900',
                                    'cursor-pointer select-none px-4 py-2'
                                  )
                                }
                              >
                                {option.label}
                              </Listbox.Option>
                            ))}
                          </Listbox.Options>
                        </div>
                      </Listbox>

                      <button
                        onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
                        className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-700 transition-colors hover:bg-slate-100"
                      >
                        {sortOrder === 'desc' ? '↓' : '↑'}
                      </button>
                    </div>
                  </div>

                  <div className="mt-4 flex flex-wrap items-center gap-3 text-xs text-slate-600">
                    <div className="flex items-center gap-2">
                      <span className="text-slate-500">Price (£m)</span>
                      <input
                        type="number"
                        placeholder="Min"
                        value={minPrice}
                        onChange={(e) => setMinPrice(e.target.value)}
                        className="w-20 rounded-lg border border-slate-200 px-2 py-1 text-xs focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-200"
                      />
                      <span className="text-slate-400">-</span>
                      <input
                        type="number"
                        placeholder="Max"
                        value={maxPrice}
                        onChange={(e) => setMaxPrice(e.target.value)}
                        className="w-20 rounded-lg border border-slate-200 px-2 py-1 text-xs focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-200"
                      />
                    </div>
                    <label className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        checked={onlyAvailable}
                        onChange={(e) => setOnlyAvailable(e.target.checked)}
                        className="h-4 w-4 rounded border-slate-300 text-blue-600 focus:ring-blue-500"
                      />
                      <span>Available only</span>
                    </label>
                  </div>
                </div>

                {/* Players Table */}
                <div className="max-h-[32rem] overflow-y-auto px-2">
                  {fetchError && (
                    <div className="rounded-2xl border border-red-200 bg-red-50/80 px-4 py-3 text-sm text-red-700">
                      {fetchError}
                    </div>
                  )}
                  {loading ? (
                    <div className="flex flex-col items-center gap-2 py-10 text-slate-500">
                      <div className="loading-spinner h-8 w-8 rounded-full border border-blue-200 border-t-blue-600"></div>
                      Loading players...
                    </div>
                  ) : filteredPlayers.length === 0 ? (
                    <div className="py-12 text-center text-sm text-slate-500">
                      No players match your filters. Try adjusting the search or removing constraints.
                    </div>
                  ) : (
                    <>
                      <div className="hidden md:block">
                        <table className="min-w-full overflow-hidden rounded-2xl border border-slate-200 bg-white text-sm">
                          <thead className="bg-slate-50 text-xs font-semibold uppercase tracking-wide text-slate-500">
                            <tr>
                              {multiSelectMode && <th className="px-3 py-3 text-left">Select</th>}
                              <th className="px-3 py-3 text-left">Player</th>
                              <th className="px-3 py-3 text-left">Position</th>
                              <th className="px-3 py-3 text-left">Team</th>
                              <th className="px-3 py-3 text-right">Price</th>
                              <th className="px-3 py-3 text-right">Total pts</th>
                              <th className="px-3 py-3 text-right">Form</th>
                              <th className="px-3 py-3 text-right">Ownership</th>
                              <th className="px-3 py-3 text-center">Status</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-slate-100">
                            {filteredPlayers.map((player) => {
                              const playerStatus = getPlayerStatus(player);
                              const isCurrentPlayer = currentPlayer?.id === player.id;
                              const isTempSelected = multiSelectMode && tempSelectedPlayers.some(p => p.id === player.id);
                              const isAlreadyInTeam = selectedPlayers.some(p => p.id === player.id);
                              const isDisabled = isAlreadyInTeam && !isCurrentPlayer && !isTempSelected;

                              return (
                                <tr
                                  key={player.id}
                                  onClick={() => {
                                    if (isDisabled) return;
                                    handlePlayerSelect(player);
                                  }}
                                  className={classNames(
                                    'cursor-pointer transition-colors',
                                    isDisabled ? 'pointer-events-none opacity-60' : 'hover:bg-blue-50/40',
                                    isTempSelected ? 'bg-green-50' : '',
                                    isCurrentPlayer ? 'bg-blue-50' : ''
                                  )}
                                >
                                  {multiSelectMode && (
                                    <td className="px-3 py-3">
                                      <input
                                        type="checkbox"
                                        checked={isTempSelected}
                                        onChange={() => {}}
                                        disabled={isDisabled}
                                        className="h-4 w-4 rounded border-slate-300 text-blue-600 focus:ring-blue-500"
                                      />
                                    </td>
                                  )}
                                  <td className="px-3 py-3">
                                    <div className="font-semibold text-slate-900">{player.web_name}</div>
                                    <div className="text-xs text-slate-500">{player.first_name} {player.second_name}</div>
                                  </td>
                                  <td className="px-3 py-3 text-slate-500">{player.position}</td>
                                  <td className="px-3 py-3 text-slate-500">{player.team_name}</td>
                                  <td className="px-3 py-3 text-right font-semibold text-slate-900">{formatPrice(player.now_cost)}</td>
                                  <td className="px-3 py-3 text-right font-semibold text-slate-900">{player.total_points}</td>
                                  <td className="px-3 py-3 text-right text-blue-600">{player.form}</td>
                                  <td className="px-3 py-3 text-right text-slate-600">{player.selected_by_percent}%</td>
                                  <td className="px-3 py-3 text-center">
                                    {playerStatus && playerStatus.status !== 'Available' ? (
                                      <span className={`rounded-full px-2 py-1 text-xs font-semibold ${playerStatus.color}`}>
                                        {playerStatus.icon}
                                      </span>
                                    ) : (
                                      <span className="text-xs font-semibold text-green-600">Available</span>
                                    )}
                                  </td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>

                      <div className="space-y-3 md:hidden">
                        {filteredPlayers.map((player) => {
                          const playerStatus = getPlayerStatus(player);
                          const isCurrentPlayer = currentPlayer?.id === player.id;
                          const isTempSelected = multiSelectMode && tempSelectedPlayers.some(p => p.id === player.id);
                          const isAlreadyInTeam = selectedPlayers.some(p => p.id === player.id);
                          const isDisabled = isAlreadyInTeam && !isCurrentPlayer && !isTempSelected;

                          return (
                            <button
                              key={player.id}
                              onClick={() => {
                                if (isDisabled) return;
                                handlePlayerSelect(player);
                              }}
                              disabled={isDisabled}
                              className={classNames(
                                'w-full rounded-2xl border border-slate-200 bg-white p-4 text-left shadow-sm transition-colors',
                                isDisabled ? 'opacity-60' : 'hover:border-blue-200',
                                isTempSelected ? 'border-green-300 bg-green-50' : '',
                                isCurrentPlayer ? 'border-blue-300 bg-blue-50' : ''
                              )}
                            >
                              <div className="flex items-center justify-between">
                                <div>
                                  <p className="text-base font-semibold text-slate-900">{player.web_name}</p>
                                  <p className="text-xs text-slate-500">{player.team_name}</p>
                                </div>
                                <div className="text-right text-sm font-semibold text-slate-900">{formatPrice(player.now_cost)}</div>
                              </div>
                              <div className="mt-2 flex items-center justify-between text-xs text-slate-500">
                                <span>{player.position}</span>
                                <span>{player.total_points} pts</span>
                                <span>{player.form} form</span>
                                <span>{player.selected_by_percent}% EO</span>
                              </div>
                              <div className="mt-2">
                                {playerStatus && playerStatus.status !== 'Available' ? (
                                  <span className={`rounded-full px-2 py-1 text-xs font-semibold ${playerStatus.color}`}>
                                    {playerStatus.status}
                                  </span>
                                ) : (
                                  <span className="rounded-full bg-green-50 px-2 py-1 text-xs font-semibold text-green-600">
                                    Available
                                  </span>
                                )}
                              </div>
                            </button>
                          );
                        })}
                      </div>
                    </>
                  )}
                </div>

                {/* Footer */}
                <div className="bg-gray-50 px-8 py-4 flex justify-between items-center border-t-2">
                  <div className="text-sm text-gray-600">
                    {filteredPlayers.length} players found
                    {multiSelectMode && tempSelectedPlayers.length > 0 && (
                      <span className="ml-4 font-semibold text-green-600">
                        • {tempSelectedPlayers.length} selected
                      </span>
                    )}
                  </div>
                  <div className="flex space-x-3">
                    {multiSelectMode && tempSelectedPlayers.length > 0 ? (
                      <>
                        <button
                          onClick={handleCancelBatch}
                          className="px-6 py-2 bg-gray-300 text-gray-700 rounded-xl hover:bg-gray-400 transition-colors font-medium"
                        >
                          Cancel
                        </button>
                        <button
                          onClick={handleConfirmBatch}
                          className="px-6 py-2 bg-green-600 text-white rounded-xl hover:bg-green-700 transition-colors font-medium shadow-lg"
                        >
                          ✓ Add {tempSelectedPlayers.length} Player{tempSelectedPlayers.length > 1 ? 's' : ''}
                        </button>
                      </>
                    ) : (
                      <button
                        onClick={onClose}
                        className="px-6 py-2 bg-gray-300 text-gray-700 rounded-xl hover:bg-gray-400 transition-colors font-medium"
                      >
                        Close
                      </button>
                    )}
                  </div>
                </div>
              </Dialog.Panel>
            </Transition.Child>
          </div>
        </div>
      </Dialog>
    </Transition>
  );
}
