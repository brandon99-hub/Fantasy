'use client';

import { useState, useEffect, Fragment } from 'react';
import { Dialog, Transition, Listbox } from '@headlessui/react';
import { XMarkIcon, ChevronUpDownIcon, CheckIcon, MagnifyingGlassIcon } from '@heroicons/react/24/outline';
import { Player } from '@/lib/types';
import { formatPrice, getPositionIcon, getPlayerStatus, classNames } from '@/lib/utils';

interface PlayerSelectionModalProps {
  isOpen: boolean;
  onClose: () => void;
  position: string;
  onPlayerSelect: (player: Player) => void;
  currentPlayer?: Player | null;
  batchMode?: boolean;
  selectedPlayers?: Player[];
}

interface Team {
  id: number;
  name: string;
  short_name: string;
}

// Simple cache for player data
const playerCache: { [position: string]: Player[] } = {};
const teamsCache: Team[] = [];

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
  const [teams, setTeams] = useState<Team[]>([]);
  const [loading, setLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedTeam, setSelectedTeam] = useState<Team | null>(null);
  const [sortBy, setSortBy] = useState<'points' | 'price' | 'form' | 'name'>('points');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  const [multiSelectMode, setMultiSelectMode] = useState(batchMode);
  const [tempSelectedPlayers, setTempSelectedPlayers] = useState<Player[]>([]);

  // Fetch players and teams when modal opens
  useEffect(() => {
    if (isOpen) {
      fetchData();
    }
  }, [isOpen, position]);

  // Reset state when modal closes
  useEffect(() => {
    if (!isOpen) {
      setPlayers([]);
      setFilteredPlayers([]);
      setTeams([]);
      setSearchQuery('');
      setSelectedTeam(null);
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
  }, [players, searchQuery, selectedTeam, sortBy, sortOrder]);

  const fetchData = async () => {
    // Map frontend position codes to database position names
    const positionMap = {
      'GKP': 'Goalkeeper',
      'DEF': 'Defender', 
      'MID': 'Midfielder',
      'FWD': 'Forward'
    };
    
    const dbPosition = positionMap[position as keyof typeof positionMap] || position;
    
    // Check cache first
    if (playerCache[dbPosition] && teamsCache.length > 0) {
      setPlayers(playerCache[dbPosition]);
      setTeams(teamsCache);
      return;
    }
    
    setLoading(true);
    try {
      const promises = [];
      
      // Fetch players only if not cached
      if (!playerCache[dbPosition]) {
        promises.push(fetch(`http://localhost:8000/api/players?position=${dbPosition}`));
      } else {
        promises.push(Promise.resolve({ ok: true, json: () => Promise.resolve(playerCache[dbPosition]) }));
      }
      
      // Fetch teams only if not cached
      if (teamsCache.length === 0) {
        promises.push(fetch('http://localhost:8000/api/teams'));
      } else {
        promises.push(Promise.resolve({ ok: true, json: () => Promise.resolve(teamsCache) }));
      }
      
      const [playersResponse, teamsResponse] = await Promise.all(promises);
      
      if (playersResponse.ok) {
        const playersData = await playersResponse.json();
        playerCache[dbPosition] = playersData || [];
        setPlayers(playersData || []);
      }

      if (teamsResponse.ok) {
        const teamsData = await teamsResponse.json();
        if (teamsCache.length === 0) {
          teamsCache.push(...(teamsData || []));
        }
        setTeams(teamsData || []);
      }
    } catch (error) {
      setPlayers([]);
      setTeams([]);
    } finally {
      setLoading(false);
    }
  };

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
    if (tempSelectedPlayers.length > 0) {
      tempSelectedPlayers.forEach((player, idx) => {
        // Use setTimeout to ensure each player is processed separately
        setTimeout(() => {
          onPlayerSelect(player);
        }, idx * 10);
      });
      setTempSelectedPlayers([]);
      // Close after all players are sent
      setTimeout(() => {
        onClose();
      }, tempSelectedPlayers.length * 10 + 100);
    }
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
                <div className="bg-gray-50 px-8 py-4 border-b">
                  <div className="flex flex-wrap items-center gap-4">
                    
                    {/* Search */}
                    <div className="flex-1 min-w-64">
                      <div className="relative">
                        <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
                        <input
                          type="text"
                          placeholder="Search players..."
                          value={searchQuery}
                          onChange={(e) => setSearchQuery(e.target.value)}
                          className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
                        />
                      </div>
                    </div>

                    {/* Team Filter */}
                    <div className="min-w-48">
                      <Listbox value={selectedTeam} onChange={setSelectedTeam}>
                        <div className="relative">
                          <Listbox.Button className="relative w-full cursor-pointer rounded-xl bg-white py-3 pl-4 pr-10 text-left border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500">
                            <span className="block truncate">
                              {selectedTeam ? selectedTeam.name : 'All Teams'}
                            </span>
                            <span className="pointer-events-none absolute inset-y-0 right-0 flex items-center pr-3">
                              <ChevronUpDownIcon className="h-5 w-5 text-gray-400" />
                            </span>
                          </Listbox.Button>
                          <Transition
                            as={Fragment}
                            leave="transition ease-in duration-100"
                            leaveFrom="opacity-100"
                            leaveTo="opacity-0"
                          >
                            <Listbox.Options className="absolute z-10 mt-1 max-h-60 w-full overflow-auto rounded-xl bg-white py-1 shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none">
                              <Listbox.Option
                                value={null}
                                className={({ active }) =>
                                  classNames(
                                    active ? 'bg-blue-50 text-blue-900' : 'text-gray-900',
                                    'relative cursor-pointer select-none py-2 px-4'
                                  )
                                }
                              >
                                All Teams
                              </Listbox.Option>
                              {teams.map((team) => (
                                <Listbox.Option
                                  key={team.id}
                                  value={team}
                                  className={({ active }) =>
                                    classNames(
                                      active ? 'bg-blue-50 text-blue-900' : 'text-gray-900',
                                      'relative cursor-pointer select-none py-2 px-4'
                                    )
                                  }
                                >
                                  {({ selected }) => (
                                    <div className="flex items-center justify-between">
                                      <span className={classNames(selected ? 'font-semibold' : 'font-normal')}>
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

                    {/* Sort */}
                    <div className="flex items-center space-x-2">
                      <Listbox value={sortBy} onChange={setSortBy}>
                        <div className="relative">
                          <Listbox.Button className="relative cursor-pointer rounded-xl bg-white py-3 pl-4 pr-10 text-left border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500 min-w-36">
                            <span className="block truncate">
                              Sort by {sortOptions.find(opt => opt.value === sortBy)?.label}
                            </span>
                            <span className="pointer-events-none absolute inset-y-0 right-0 flex items-center pr-3">
                              <ChevronUpDownIcon className="h-5 w-5 text-gray-400" />
                            </span>
                          </Listbox.Button>
                          <Listbox.Options className="absolute z-10 mt-1 max-h-60 w-full overflow-auto rounded-xl bg-white py-1 shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none">
                            {sortOptions.map((option) => (
                              <Listbox.Option
                                key={option.value}
                                value={option.value}
                                className={({ active }) =>
                                  classNames(
                                    active ? 'bg-blue-50 text-blue-900' : 'text-gray-900',
                                    'relative cursor-pointer select-none py-2 px-4'
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
                        className="px-4 py-3 bg-white border border-gray-300 rounded-xl hover:bg-gray-50 transition-colors"
                      >
                        {sortOrder === 'desc' ? '↓' : '↑'}
                      </button>
                    </div>
                  </div>
                </div>

                {/* Players Table */}
                <div className="max-h-96 overflow-y-auto">
                  {loading ? (
                    <div className="flex items-center justify-center py-12">
                      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                      <span className="ml-3 text-gray-600">Loading players...</span>
                    </div>
                  ) : filteredPlayers.length === 0 ? (
                    <div className="text-center py-12">
                      <div className="text-gray-500 text-lg">No players found</div>
                      <p className="text-gray-400 text-sm mt-2">Try adjusting your filters</p>
                    </div>
                  ) : (
                    <div className="overflow-hidden">
                      {/* Table Header */}
                      <div className="bg-gray-50 border-b border-gray-200 px-6 py-3">
                        <div className="grid grid-cols-12 gap-4 text-xs font-medium text-gray-500 uppercase tracking-wider">
                          <div className="col-span-1"></div> {/* Photo */}
                          <div className="col-span-3">Player</div>
                          <div className="col-span-2">Team</div>
                          <div className="col-span-1 text-center">Price</div>
                          <div className="col-span-1 text-center">Points</div>
                          <div className="col-span-1 text-center">Form</div>
                          <div className="col-span-2 text-center">Selected</div>
                          <div className="col-span-1 text-center">Status</div>
                        </div>
                      </div>
                      
                      {/* Table Body */}
                      <div className="divide-y divide-gray-100">
                        {filteredPlayers.map((player) => {
                          const playerStatus = getPlayerStatus(player);
                          const isCurrentPlayer = currentPlayer?.id === player.id;
                          const isTempSelected = multiSelectMode && tempSelectedPlayers.some(p => p.id === player.id);
                          const isAlreadyInTeam = selectedPlayers.some(p => p.id === player.id);
                          
                          return (
                            <button
                              key={player.id}
                              onClick={() => handlePlayerSelect(player)}
                              disabled={isAlreadyInTeam && !isCurrentPlayer}
                              className={classNames(
                                'w-full px-6 py-4 text-left transition-colors',
                                isTempSelected ? 'bg-green-100 border-l-4 border-green-500' : '',
                                isCurrentPlayer ? 'bg-blue-100 border-l-4 border-blue-500' : '',
                                isAlreadyInTeam && !isCurrentPlayer ? 'opacity-50 cursor-not-allowed bg-gray-100' : 'hover:bg-blue-50',
                                !isTempSelected && !isCurrentPlayer && !isAlreadyInTeam ? '' : ''
                              )}
                            >
                              <div className="grid grid-cols-12 gap-4 items-center">
                                {/* Player Photo */}
                                <div className="col-span-1">
                                  <div className={`w-10 h-10 rounded-full flex items-center justify-center text-sm font-bold ${getPositionColor(player.position)}`}>
                                    {player.photo ? (
                                      <img 
                                        src={`https://resources.premierleague.com/premierleague/photos/players/250x250/p${player.code}.png`}
                                        alt={player.web_name}
                                        className="w-10 h-10 rounded-full object-cover"
                                        onError={(e) => {
                                          e.currentTarget.style.display = 'none';
                                          e.currentTarget.nextElementSibling.style.display = 'flex';
                                        }}
                                      />
                                    ) : null}
                                    <span className={player.photo ? 'hidden' : 'block'}>
                                      {getPositionIcon(player.position)}
                                    </span>
                                  </div>
                                </div>
                                
                                {/* Player Name */}
                                <div className="col-span-3">
                                  <div className="flex items-center space-x-2">
                                    {multiSelectMode && (
                                      <input
                                        type="checkbox"
                                        checked={isTempSelected}
                                        onChange={() => {}}
                                        className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
                                      />
                                    )}
                                    <div className="font-semibold text-gray-900">
                                      {player.web_name}
                                    </div>
                                    {isTempSelected && (
                                      <span className="bg-green-500 text-white text-xs px-2 py-1 rounded-full">
                                        ✓ Selected
                                      </span>
                                    )}
                                    {isCurrentPlayer && (
                                      <span className="bg-blue-500 text-white text-xs px-2 py-1 rounded-full">
                                        Current
                                      </span>
                                    )}
                                    {isAlreadyInTeam && !isCurrentPlayer && (
                                      <span className="bg-gray-400 text-white text-xs px-2 py-1 rounded-full">
                                        In Team
                                      </span>
                                    )}
                                  </div>
                                  <div className="text-sm text-gray-500">
                                    {player.first_name} {player.second_name}
                                  </div>
                                </div>
                                
                                {/* Team */}
                                <div className="col-span-2">
                                  <div className="font-medium text-gray-900">
                                    {player.team_name}
                                  </div>
                                </div>
                                
                                {/* Price */}
                                <div className="col-span-1 text-center">
                                  <div className="text-lg font-bold text-green-600">
                                    {formatPrice(player.now_cost)}
                                  </div>
                                </div>
                                
                                {/* Points */}
                                <div className="col-span-1 text-center">
                                  <div className="font-semibold text-gray-900">
                                    {player.total_points}
                                  </div>
                                </div>
                                
                                {/* Form */}
                                <div className="col-span-1 text-center">
                                  <div className="font-semibold text-blue-600">
                                    {player.form}
                                  </div>
                                </div>
                                
                                {/* Selected By */}
                                <div className="col-span-2 text-center">
                                  <div className="text-sm text-gray-600">
                                    {player.selected_by_percent}%
                                  </div>
                                </div>
                                
                                {/* Status */}
                                <div className="col-span-1 text-center">
                                  {playerStatus && playerStatus.status !== 'Available' ? (
                                    <span className={`text-xs px-2 py-1 rounded-full ${playerStatus.color}`}>
                                      {playerStatus.icon}
                                    </span>
                                  ) : (
                                    <span className="text-green-500 text-xs">✓</span>
                                  )}
                                </div>
                              </div>
                            </button>
                          );
                        })}
                      </div>
                    </div>
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
