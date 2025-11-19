'use client';

import { useState } from 'react';
import { Player } from '@/lib/types';
import { getPositionIcon, formatPrice } from '@/lib/utils';
import PlayerSlot from './PlayerSlot';

interface FormationLayoutProps {
  players: Player[];
  onPlayerChange: (player: Player | null, position: string, index: number) => void;
  startingXI?: number[];
  onToggleStartingXI?: (playerId: number) => void;
}

export default function FormationLayout({ 
  players, 
  onPlayerChange, 
  startingXI = [], 
  onToggleStartingXI 
}: FormationLayoutProps) {
  const getPlayersByPosition = (position: string) => {
    // Map frontend position codes to database position names
    const positionMap = {
      'GKP': 'Goalkeeper',
      'DEF': 'Defender', 
      'MID': 'Midfielder',
      'FWD': 'Forward'
    };
    
    const dbPosition = positionMap[position as keyof typeof positionMap] || position;
    return players.filter(p => p.position === dbPosition);
  };

  const getPlayerForSlot = (position: string, index: number): Player | null => {
    const positionPlayers = getPlayersByPosition(position);
    return positionPlayers[index] || null;
  };

  return (
    <div className="relative w-full max-w-5xl mx-auto">
      {/* Football Pitch Container */}
      <div className="formation-field rounded-3xl p-8 min-h-[800px] relative shadow-2xl border-4 border-white/20" 
           style={{
             background: 'linear-gradient(135deg, #2d5a27 0%, #4a7c59 25%, #2d5a27 50%, #4a7c59 75%, #2d5a27 100%)',
             backgroundSize: '60px 60px'
           }}>
        
        {/* Grass Pattern Overlay */}
        <div className="absolute inset-0 opacity-10 pointer-events-none"
             style={{
               backgroundImage: `repeating-linear-gradient(
                 0deg,
                 transparent,
                 transparent 35px,
                 rgba(255,255,255,0.1) 35px,
                 rgba(255,255,255,0.1) 40px
               )`
             }}>
        </div>

        {/* Enhanced Field Lines */}
        <div className="absolute inset-4 border-4 border-white/50 rounded-xl pointer-events-none">
          
          {/* Center Circle */}
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-48 h-48 border-4 border-white/50 rounded-full"></div>
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-3 h-3 bg-white/70 rounded-full"></div>
          
          {/* Center Line */}
          <div className="absolute top-0 bottom-0 left-1/2 transform -translate-x-1/2 w-1 bg-white/50"></div>
          
          {/* Penalty Areas */}
          <div className="absolute top-0 left-1/2 transform -translate-x-1/2 w-44 h-24 border-4 border-white/50 border-t-0 rounded-b-2xl"></div>
          <div className="absolute bottom-0 left-1/2 transform -translate-x-1/2 w-44 h-24 border-4 border-white/50 border-b-0 rounded-t-2xl"></div>
          
          {/* Goal Areas */}
          <div className="absolute top-0 left-1/2 transform -translate-x-1/2 w-20 h-12 border-4 border-white/50 border-t-0 rounded-b-lg"></div>
          <div className="absolute bottom-0 left-1/2 transform -translate-x-1/2 w-20 h-12 border-4 border-white/50 border-b-0 rounded-t-lg"></div>
          
          {/* Penalty Spots */}
          <div className="absolute top-16 left-1/2 transform -translate-x-1/2 w-2 h-2 bg-white/70 rounded-full"></div>
          <div className="absolute bottom-16 left-1/2 transform -translate-x-1/2 w-2 h-2 bg-white/70 rounded-full"></div>
          
          {/* Corner Arcs */}
          <div className="absolute -top-1 -left-1 w-12 h-12 border-r-4 border-b-4 border-white/50 rounded-br-full"></div>
          <div className="absolute -top-1 -right-1 w-12 h-12 border-l-4 border-b-4 border-white/50 rounded-bl-full"></div>
          <div className="absolute -bottom-1 -left-1 w-12 h-12 border-r-4 border-t-4 border-white/50 rounded-tr-full"></div>
          <div className="absolute -bottom-1 -right-1 w-12 h-12 border-l-4 border-t-4 border-white/50 rounded-tl-full"></div>
          
          {/* Goal Posts */}
          <div className="absolute top-0 left-1/2 transform -translate-x-1/2 -translate-y-1 w-16 h-2 bg-white/80 rounded-full shadow-lg"></div>
          <div className="absolute bottom-0 left-1/2 transform -translate-x-1/2 translate-y-1 w-16 h-2 bg-white/80 rounded-full shadow-lg"></div>
        </div>

      {/* Formation Layout */}
      <div className="relative z-10 h-full flex flex-col justify-between">
        
        {/* Forwards - Attack */}
        <div className="flex justify-center items-center py-6">
          <div className="flex space-x-12">
            {Array.from({ length: 3 }, (_, index) => (
              <div key={`FWD-${index}`} className="transform hover:scale-105 transition-transform duration-200">
                <PlayerSlot
                  player={getPlayerForSlot('FWD', index)}
                  position="FWD"
                  index={index}
                  onPlayerChange={onPlayerChange}
                  allSelectedPlayers={players}
                  startingXI={startingXI}
                  onToggleStartingXI={onToggleStartingXI}
                />
              </div>
            ))}
          </div>
        </div>

        {/* Midfielders - Midfield */}
        <div className="flex justify-center items-center py-6">
          <div className="flex space-x-8">
            {Array.from({ length: 5 }, (_, index) => (
              <div key={`MID-${index}`} className="transform hover:scale-105 transition-transform duration-200">
                <PlayerSlot
                  player={getPlayerForSlot('MID', index)}
                  position="MID"
                  index={index}
                  onPlayerChange={onPlayerChange}
                  allSelectedPlayers={players}
                  startingXI={startingXI}
                  onToggleStartingXI={onToggleStartingXI}
                />
              </div>
            ))}
          </div>
        </div>

        {/* Defenders - Defense */}
        <div className="flex justify-center items-center py-6">
          <div className="flex space-x-8">
            {Array.from({ length: 5 }, (_, index) => (
              <div key={`DEF-${index}`} className="transform hover:scale-105 transition-transform duration-200">
                <PlayerSlot
                  player={getPlayerForSlot('DEF', index)}
                  position="DEF"
                  index={index}
                  onPlayerChange={onPlayerChange}
                  allSelectedPlayers={players}
                  startingXI={startingXI}
                  onToggleStartingXI={onToggleStartingXI}
                />
              </div>
            ))}
          </div>
        </div>

        {/* Goalkeepers - Goal */}
        <div className="flex justify-center items-center py-6">
          <div className="flex space-x-16">
            {Array.from({ length: 2 }, (_, index) => (
              <div key={`GKP-${index}`} className="transform hover:scale-105 transition-transform duration-200">
                <PlayerSlot
                  player={getPlayerForSlot('GKP', index)}
                  position="GKP"
                  index={index}
                  onPlayerChange={onPlayerChange}
                  allSelectedPlayers={players}
                  startingXI={startingXI}
                  onToggleStartingXI={onToggleStartingXI}
                />
              </div>
            ))}
          </div>
        </div>

        </div>

        {/* Enhanced Position Labels */}
        <div className="absolute top-8 left-8 space-y-3">
          {[
            { pos: 'FWD', label: 'Forwards', count: 3, color: 'text-red-100', bgColor: 'bg-red-500/30' },
            { pos: 'MID', label: 'Midfielders', count: 5, color: 'text-blue-100', bgColor: 'bg-blue-500/30' },
            { pos: 'DEF', label: 'Defenders', count: 5, color: 'text-yellow-100', bgColor: 'bg-yellow-500/30' },
            { pos: 'GKP', label: 'Goalkeepers', count: 2, color: 'text-green-100', bgColor: 'bg-green-500/30' }
          ].map(({ pos, label, count, color, bgColor }) => {
            const positionPlayers = players.filter(p => p.position === pos);
            return (
              <div key={pos} className={`${bgColor} backdrop-blur-sm rounded-xl px-4 py-3 border border-white/20 shadow-lg`}>
                <div className={`flex items-center space-x-3 text-sm font-semibold ${color}`}>
                  <span className="text-xl">{getPositionIcon(pos)}</span>
                  <span className="flex-1">{label}</span>
                  <span className="text-xs bg-white/40 text-gray-800 px-3 py-1 rounded-full font-bold">
                    {positionPlayers.length}/{count}
                  </span>
                </div>
              </div>
            );
          })}
        </div>

        {/* Enhanced Team Stats */}
        <div className="absolute top-8 right-8 bg-gradient-to-br from-white/25 to-white/10 backdrop-blur-md rounded-2xl p-6 text-white border border-white/20 shadow-2xl min-w-[200px]">
          <div className="text-center mb-4">
            <div className="text-2xl font-bold bg-gradient-to-r from-white to-blue-100 bg-clip-text text-transparent">
              {players.length}/15
            </div>
            <div className="text-xs opacity-90 font-medium">Players Selected</div>
          </div>
          
          <div className="space-y-3 text-sm">
            <div className="flex justify-between items-center p-2 bg-white/10 rounded-lg">
              <span className="opacity-90 font-medium">💰 Value:</span>
              <span className="font-bold text-green-200">
                {formatPrice(players.reduce((sum, p) => sum + p.now_cost, 0))}
              </span>
            </div>
            
            <div className="flex justify-between items-center p-2 bg-white/10 rounded-lg">
              <span className="opacity-90 font-medium">💳 Remaining:</span>
              <span className="font-bold text-yellow-200">
                {formatPrice(1000 - players.reduce((sum, p) => sum + p.now_cost, 0))}
              </span>
            </div>
            
            {players.length > 0 && (
              <div className="pt-2 border-t border-white/20">
                <div className="flex justify-between items-center p-2 bg-white/10 rounded-lg">
                  <span className="opacity-90 font-medium">⚡ Avg Points:</span>
                  <span className="font-bold text-blue-200">
                    {(players.reduce((sum, p) => sum + p.total_points, 0) / players.length).toFixed(1)}
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
