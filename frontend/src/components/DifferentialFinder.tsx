/**
 * Differential Finder Component
 * Finds and displays low-owned high-value players with risk ratings
 */

'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface Differential {
    web_name: string;
    team_name: string;
    position: string;
    now_cost: number;
    predicted_points: number;
    selected_by_percent: number;
    differential_score: number;
    risk_rating: string;
    form: number;
    recommendation_reason: string;
}

export default function DifferentialFinder() {
    const [ownershipThreshold, setOwnershipThreshold] = useState(10);
    const [riskTolerance, setRiskTolerance] = useState<'low' | 'medium' | 'high'>('medium');
    const [position, setPosition] = useState<string>('');

    const { data, isLoading } = useQuery<Differential[]>({
        queryKey: ['differentials', ownershipThreshold, riskTolerance, position],
        queryFn: async () => {
            const params = new URLSearchParams({
                ownership_threshold: ownershipThreshold.toString(),
                risk_tolerance: riskTolerance,
                ...(position && { position }),
            });
            const res = await axios.get(`${API_BASE}/api/advanced/differentials?${params}`);
            return res.data;
        },
    });

    const getRiskColor = (rating: string) => {
        if (rating.includes('Low')) return 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300';
        if (rating.includes('Medium')) return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300';
        return 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300';
    };

    return (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
            {/* Header */}
            <div className="mb-6">
                <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
                    Differential Finder
                </h2>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                    Find low-owned players with high potential
                </p>
            </div>

            {/* Filters */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Max Ownership
                    </label>
                    <input
                        type="range"
                        min="1"
                        max="20"
                        value={ownershipThreshold}
                        onChange={(e) => setOwnershipThreshold(Number(e.target.value))}
                        className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
                    />
                    <div className="text-center text-sm text-gray-600 dark:text-gray-400 mt-1">
                        {ownershipThreshold}%
                    </div>
                </div>

                <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Risk Tolerance
                    </label>
                    <select
                        value={riskTolerance}
                        onChange={(e) => setRiskTolerance(e.target.value as any)}
                        className="w-full px-3 py-2 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg text-gray-900 dark:text-white"
                    >
                        <option value="low">Low Risk</option>
                        <option value="medium">Medium Risk</option>
                        <option value="high">High Risk</option>
                    </select>
                </div>

                <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Position
                    </label>
                    <select
                        value={position}
                        onChange={(e) => setPosition(e.target.value)}
                        className="w-full px-3 py-2 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg text-gray-900 dark:text-white"
                    >
                        <option value="">All Positions</option>
                        <option value="GKP">Goalkeeper</option>
                        <option value="DEF">Defender</option>
                        <option value="MID">Midfielder</option>
                        <option value="FWD">Forward</option>
                    </select>
                </div>
            </div>

            {/* Results */}
            {isLoading ? (
                <div className="space-y-3">
                    {[1, 2, 3].map((i) => (
                        <div key={i} className="animate-pulse bg-gray-100 dark:bg-gray-700 rounded-lg h-24"></div>
                    ))}
                </div>
            ) : (
                <div className="space-y-3">
                    {data?.map((player, idx) => (
                        <div
                            key={idx}
                            className="bg-gradient-to-r from-gray-50 to-white dark:from-gray-700 dark:to-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg p-4 hover:shadow-md transition-shadow"
                        >
                            <div className="flex items-start justify-between">
                                <div className="flex-1">
                                    <div className="flex items-center gap-3 mb-2">
                                        <h3 className="text-lg font-bold text-gray-900 dark:text-white">
                                            {player.web_name}
                                        </h3>
                                        <span className={`px-2 py-1 rounded text-xs font-semibold ${getRiskColor(player.risk_rating)}`}>
                                            {player.risk_rating}
                                        </span>
                                    </div>

                                    <div className="flex items-center gap-4 text-sm text-gray-600 dark:text-gray-400 mb-2">
                                        <span>{player.team_name}</span>
                                        <span>•</span>
                                        <span>{player.position}</span>
                                        <span>•</span>
                                        <span>£{(player.now_cost / 10).toFixed(1)}m</span>
                                    </div>

                                    <div className="text-sm text-gray-700 dark:text-gray-300">
                                        {player.recommendation_reason}
                                    </div>
                                </div>

                                <div className="text-right ml-4">
                                    <div className="text-sm text-gray-500 dark:text-gray-400">Ownership</div>
                                    <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                                        {player.selected_by_percent.toFixed(1)}%
                                    </div>
                                    <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                                        {player.predicted_points.toFixed(1)} pts
                                    </div>
                                </div>
                            </div>

                            {/* Score Bar */}
                            <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-600">
                                <div className="flex items-center gap-2">
                                    <span className="text-xs text-gray-500 dark:text-gray-400">Differential Score:</span>
                                    <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                                        <div
                                            className="bg-gradient-to-r from-blue-500 to-purple-500 h-2 rounded-full"
                                            style={{ width: `${Math.min(player.differential_score * 10, 100)}%` }}
                                        ></div>
                                    </div>
                                    <span className="text-xs font-semibold text-gray-700 dark:text-gray-300">
                                        {player.differential_score.toFixed(1)}
                                    </span>
                                </div>
                            </div>
                        </div>
                    ))}

                    {data?.length === 0 && (
                        <div className="text-center py-12 text-gray-500 dark:text-gray-400">
                            No differentials found with current filters
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
