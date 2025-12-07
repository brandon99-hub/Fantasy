/**
 * Injury Risk Dashboard Component
 * Displays injury risk for all players with filtering
 */

'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface InjuryRisk {
    web_name: string;
    team_name: string;
    injury_risk_score: number;
    injury_risk_level: string;
    injury_recommendation: string;
    injury_risk_factors: string;
}

export default function InjuryRiskDashboard() {
    const [minRisk, setMinRisk] = useState(0.3);

    const { data, isLoading } = useQuery<InjuryRisk[]>({
        queryKey: ['injury-risk-all', minRisk],
        queryFn: async () => {
            const res = await axios.get(`${API_BASE}/api/features/injury-risk-all?min_risk=${minRisk}`);
            return res.data;
        },
    });

    const getRiskIcon = (level: string) => {
        if (level.includes('Low')) return '🟢';
        if (level.includes('Medium')) return '🟡';
        if (level.includes('High')) return '🟠';
        return '🔴';
    };

    const getRiskColor = (level: string) => {
        if (level.includes('Low')) return 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800';
        if (level.includes('Medium')) return 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800';
        if (level.includes('High')) return 'bg-orange-50 dark:bg-orange-900/20 border-orange-200 dark:border-orange-800';
        return 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800';
    };

    return (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
            {/* Header */}
            <div className="mb-6">
                <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
                    Injury Risk Monitor
                </h2>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                    Players at risk based on workload and fatigue analysis
                </p>
            </div>

            {/* Filter */}
            <div className="mb-6">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Minimum Risk Level
                </label>
                <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.1"
                    value={minRisk}
                    onChange={(e) => setMinRisk(Number(e.target.value))}
                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
                />
                <div className="flex justify-between text-xs text-gray-600 dark:text-gray-400 mt-1">
                    <span>Low</span>
                    <span className="font-semibold">{(minRisk * 100).toFixed(0)}%</span>
                    <span>High</span>
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
                            className={`border rounded-lg p-4 ${getRiskColor(player.injury_risk_level)}`}
                        >
                            <div className="flex items-start justify-between mb-3">
                                <div className="flex items-center gap-3">
                                    <span className="text-2xl">{getRiskIcon(player.injury_risk_level)}</span>
                                    <div>
                                        <h3 className="font-bold text-gray-900 dark:text-white">
                                            {player.web_name}
                                        </h3>
                                        <p className="text-sm text-gray-600 dark:text-gray-400">
                                            {player.team_name}
                                        </p>
                                    </div>
                                </div>

                                <div className="text-right">
                                    <div className="text-sm font-semibold text-gray-700 dark:text-gray-300">
                                        {player.injury_risk_level}
                                    </div>
                                    <div className="text-xs text-gray-600 dark:text-gray-400">
                                        {(player.injury_risk_score * 100).toFixed(0)}% risk
                                    </div>
                                </div>
                            </div>

                            <div className="bg-white dark:bg-gray-800 rounded p-3 mb-2">
                                <div className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-1">
                                    Recommendation:
                                </div>
                                <div className="text-sm text-gray-600 dark:text-gray-400">
                                    {player.injury_recommendation}
                                </div>
                            </div>

                            <div className="text-xs text-gray-600 dark:text-gray-400">
                                <span className="font-semibold">Risk Factors:</span> {player.injury_risk_factors}
                            </div>
                        </div>
                    ))}

                    {data?.length === 0 && (
                        <div className="text-center py-12 text-gray-500 dark:text-gray-400">
                            No high-risk players found
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
