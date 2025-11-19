'use client';

import { useState, useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import { Player } from '@/lib/types';
import { formatPrice, getPositionIcon, getPlayerStatus, classNames, getTeamColor } from '@/lib/utils';
import PlayerSelectionModal from './PlayerSelectionModal';

interface PlayerSlotProps {
  player: Player | null;
  position: string;
  index: number;
  onPlayerChange: (player: Player | null, position: string, index: number) => void;
  allSelectedPlayers?: Player[];
  startingXI?: number[];
  onToggleStartingXI?: (playerId: number) => void;
}

export default function PlayerSlot({ 
  player, 
  position, 
  index, 
  onPlayerChange, 
  allSelectedPlayers = [],
  startingXI = [],
  onToggleStartingXI
}: PlayerSlotProps) {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isReplaceMode, setIsReplaceMode] = useState(false);
  const [showContextMenu, setShowContextMenu] = useState(false);
  const [contextMenuPos, setContextMenuPos] = useState({ x: 0, y: 0 });
  const contextMenuRef = useRef<HTMLDivElement>(null);
  
  const isInStartingXI = player ? startingXI.includes(player.id) : false;
  
  // Close context menu when clicking outside
  useEffect(() => {
    if (!showContextMenu) return;
    
    const handleClickOutside = (event: MouseEvent) => {
      if (contextMenuRef.current && !contextMenuRef.current.contains(event.target as Node)) {
        setShowContextMenu(false);
      }
    };
    
    // Use a small delay to avoid immediate closure
    const timer = setTimeout(() => {
      document.addEventListener('click', handleClickOutside, true);
    }, 100);
    
    return () => {
      clearTimeout(timer);
      document.removeEventListener('click', handleClickOutside, true);
    };
  }, [showContextMenu]);

  const handlePlayerSelect = (selectedPlayer: Player) => {
    onPlayerChange(selectedPlayer, position, index);
    setIsModalOpen(false);
    setIsReplaceMode(false);
  };

  const handleRemovePlayer = () => {
    onPlayerChange(null, position, index);
  };

  const handleReplaceClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    setIsReplaceMode(true);
    setIsModalOpen(true);
    setShowContextMenu(false);
  };
  
  const handleContextMenu = (e: React.MouseEvent) => {
    if (player && onToggleStartingXI) {
      e.preventDefault();
      e.stopPropagation();
      
      // Use exact click position
      setContextMenuPos({ x: e.clientX, y: e.clientY });
      setShowContextMenu(true);
    }
  };
  
  const handleToggleStartingXI = () => {
    if (player && onToggleStartingXI) {
      onToggleStartingXI(player.id);
    }
    setShowContextMenu(false);
  };

  const playerStatus = player ? getPlayerStatus(player) : null;

  return (
    <>
      <div className="player-slot relative">
        <button
          className={classNames(
            "relative w-20 h-28 cursor-pointer focus:outline-none player-slot group",
            player
              ? "transform hover:scale-105 transition-transform duration-150"
              : "opacity-80 hover:opacity-100",
            isInStartingXI && "ring-4 ring-green-400 rounded-lg"
          )}
          onClick={(e) => {
            if (!showContextMenu) {
              setIsModalOpen(true);
            }
          }}
          onContextMenu={handleContextMenu}
        >
          {/* Starting XI Badge */}
          {isInStartingXI && (
            <div className="absolute -top-2 -right-2 z-10 bg-green-500 text-white rounded-full w-6 h-6 flex items-center justify-center text-xs font-bold shadow-lg">
              XI
            </div>
          )}
          {!isInStartingXI && player && (
            <div className="absolute -top-2 -right-2 z-10 bg-gray-400 text-white rounded-full w-6 h-6 flex items-center justify-center text-xs font-bold shadow-lg">
              B
            </div>
          )}
          {player ? (
            <div className="relative">
              {/* Real Team Jersey */}
              <div className="w-20 h-24 mx-auto mb-1 relative">
                {/* Jersey Image */}
                <div className="relative w-full h-20">
                  <img 
                    src={`https://resources.premierleague.com/premierleague/photos/players/250x250/p${player.code}.png`}
                    alt={player.web_name}
                    className="w-full h-full object-cover rounded-lg shadow-lg border-2 border-white"
                    onError={(e) => {
                      const target = e.currentTarget;
                      if (target && !target.dataset.fallbackAttempted) {
                        // First fallback: team jersey template
                        target.dataset.fallbackAttempted = 'true';
                        target.src = `https://fantasy.premierleague.com/dist/img/shirts/standard/shirt_${player.team}_1-110.png`;
                        target.onerror = () => {
                          // Final fallback: hide image and show colored jersey
                          const fallback = target.nextElementSibling as HTMLElement;
                          if (target && fallback) {
                            target.style.display = 'none';
                            fallback.style.display = 'block';
                          }
                        };
                      }
                    }}
                  />
                  
                  {/* Fallback Jersey */}
                  <div 
                    className="hidden w-full h-full rounded-lg shadow-lg border-2 border-white relative overflow-hidden"
                    style={{
                      background: `linear-gradient(135deg, ${getTeamColor(player.team_name)} 0%, ${getTeamColor(player.team_name)}dd 100%)`
                    }}
                  >
                    <div className="absolute inset-0 flex items-center justify-center">
                      <div className="w-12 h-12 bg-white/20 rounded-full flex items-center justify-center">
                        <span className="text-white text-lg font-bold">
                          {player.web_name.charAt(0)}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
                
                {/* Price Badge */}
                <div className="absolute -top-2 -right-2 bg-red-600 text-white text-xs px-2 py-1 rounded-full font-bold shadow-md z-10">
                  {formatPrice(player.now_cost)}
                </div>
                
                {/* Status Badge */}
                {playerStatus && playerStatus.status !== 'Available' && (
                  <div className="absolute -top-2 -left-2 bg-yellow-500 text-white text-xs px-1 py-1 rounded-full z-10">
                    !
                  </div>
                )}
              </div>
              
              {/* Player Name Plate */}
              <div className="bg-white rounded-md px-2 py-1 text-center shadow-lg border border-gray-200">
                <div className="text-xs font-bold text-gray-900 truncate">
                  {player.web_name}
                </div>
                <div className="text-xs text-gray-500 truncate">
                  {player.team_name.split(' ').pop()} ({position})
                </div>
              </div>
            </div>
          ) : (
            <div className="w-20 h-28 flex flex-col items-center justify-center">
              {/* Empty Slot */}
              <div className="w-20 h-20 border-4 border-dashed border-white/40 rounded-lg flex items-center justify-center mb-1 group-hover:border-white/60 transition-colors">
                <div className="text-3xl text-white/50 group-hover:text-white/70 transition-colors">
                  {getPositionIcon(position)}
                </div>
              </div>
              <div className="bg-white/20 backdrop-blur-sm rounded px-2 py-1 text-center">
                <div className="text-xs text-white/70 font-medium">
                  {position}
                </div>
              </div>
            </div>
          )}
        </button>

        {/* Remove and Replace buttons for selected players */}
        {player && (
          <div className="absolute -top-2 left-1/2 transform -translate-x-1/2 flex space-x-1 opacity-0 group-hover:opacity-100 transition-opacity">
            <button
              onClick={handleReplaceClick}
              className="bg-blue-500 hover:bg-blue-600 text-white rounded-full w-7 h-7 flex items-center justify-center text-xs transition-colors shadow-lg"
              title="Replace player"
            >
              ↻
            </button>
            <button
              onClick={(e) => {
                e.stopPropagation();
                handleRemovePlayer();
              }}
              className="bg-red-500 hover:bg-red-600 text-white rounded-full w-7 h-7 flex items-center justify-center text-xs transition-colors shadow-lg"
              title="Remove player"
            >
              ×
            </button>
          </div>
        )}
      </div>

      {/* Context Menu - Rendered via Portal to avoid z-index issues */}
      {showContextMenu && typeof document !== 'undefined' && createPortal(
        <div 
          ref={contextMenuRef}
          className="fixed z-[99999] bg-white rounded-lg shadow-2xl border-2 border-gray-300 py-1 min-w-[180px] select-none"
          style={{ 
            left: `${contextMenuPos.x}px`, 
            top: `${contextMenuPos.y}px`,
            boxShadow: '0 10px 40px rgba(0, 0, 0, 0.3)',
            pointerEvents: 'auto'
          }}
          onMouseDown={(e) => e.stopPropagation()}
          onClick={(e) => e.stopPropagation()}
        >
          <button
            onMouseDown={(e) => e.stopPropagation()}
            onClick={(e) => {
              e.preventDefault();
              e.stopPropagation();
              handleReplaceClick(e);
            }}
            className="w-full px-4 py-3 text-left text-sm hover:bg-blue-50 active:bg-blue-100 flex items-center space-x-3 text-gray-700 cursor-pointer"
          >
            <span className="text-blue-600 text-lg">↻</span>
            <span className="font-medium">Replace Player</span>
          </button>
          <div className="border-t border-gray-200" />
          <button
            onMouseDown={(e) => e.stopPropagation()}
            onClick={(e) => {
              e.preventDefault();
              e.stopPropagation();
              handleToggleStartingXI();
            }}
            className="w-full px-4 py-3 text-left text-sm hover:bg-green-50 active:bg-green-100 flex items-center space-x-3 text-gray-700 cursor-pointer"
          >
            {isInStartingXI ? (
              <>
                <span className="text-gray-600 text-lg font-bold">B</span>
                <span className="font-medium">Move to Bench</span>
              </>
            ) : (
              <>
                <span className="text-green-600 text-lg font-bold">XI</span>
                <span className="font-medium">Add to Starting XI</span>
              </>
            )}
          </button>
        </div>,
        document.body
      )}

      {/* Player Selection Modal */}
      <PlayerSelectionModal
        isOpen={isModalOpen}
        onClose={() => {
          setIsModalOpen(false);
          setIsReplaceMode(false);
        }}
        position={position}
        onPlayerSelect={handlePlayerSelect}
        currentPlayer={player}
        selectedPlayers={allSelectedPlayers}
      />
    </>
  );
}
