/**
 * Multi-Gameweek Transfer Planner Component
 * Plans transfers across multiple gameweeks with price change timing
 */

'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface TransferPlan {
    gameweeks: Array<{
        gameweek: number;
        transfers_in: Array<{ web_name: string; now_cost: number }>;
        transfers_out: Array<{ web_name: string; now_cost: number }>;
        transfer_cost: number;
        expected_points: number;
        should_transfer: boolean;
        free_transfers_before: number;
        free_transfers_after: number;
        recommendation: string;
        price_change_risk: {
            rise_soon: Array<{ player: string; probability: number }>;
            fall_soon: Array<{ player: string; probability: number }>;
        };
    }>;
    total_expected_points: number;
    total_transfer_cost: number;
    summary: string;
}

export default function MultiGWPlanner({ currentTeam }: { currentTeam: number[] }) {
    const [horizon, setHorizon] = useState(5);
    const [freeTransfers, setFreeTransfers] = useState(1);
    const [bank, setBank] = useState(0);

    const { data, isLoading, refetch } = useQuery<TransferPlan>({
        queryKey: ['multi-gw-plan', currentTeam, horizon, freeTransfers, bank],
        queryFn: async () => {
            const res = await axios.post(`${API_BASE}/api/advanced/multi-gw-plan`, {
                current_team: currentTeam,
                horizon,
                free_transfers: freeTransfers,
                bank,
            });
            return res.data;
        },
        enabled: currentTeam.length === 15,
    });

    if (currentTeam.length !== 15) {
        return (
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-4">
                <p className="text-yellow-800 dark:text-yellow-200">
                    Please select a full team of 15 players to see transfer recommendations
                </p>
            </div>
        );
    }

    return (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
            {/* Header */}
            <div className="mb-6">
                <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
                    Multi-Gameweek Transfer Planner
                </h2>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                    Plan your transfers across the next {horizon} gameweeks
                </p>
            </div>

            {/* Controls */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Planning Horizon
                    </label>
                    <select
                        value={horizon}
                        onChange={(e) => setHorizon(Number(e.target.value))}
                        className="w-full px-3 py-2 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg"
                    >
                        {[3, 4, 5, 6, 7, 8].map((h) => (
                            <option key={h} value={h}>{h} Gameweeks</option>
                        ))}
                    </select>
                </div>

                <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Free Transfers
                    </label>
                    <select
                        value={freeTransfers}
                        onChange={(e) => setFreeTransfers(Number(e.target.value))}
                        className="w-full px-3 py-2 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg"
                    >
                        <option value={0}>0 FT</option>
                        <option value={1}>1 FT</option>
                        <option value={2}>2 FT</option>
                    </select>
                </div>

                <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Bank (£m)
                    </label>
                    <input
                        type="number"
                        step="0.1"
                        value={bank}
                        onChange={(e) => setBank(Number(e.target.value))}
                        className="w-full px-3 py-2 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg"
                    />
                </div>
            </div>

            {/* Summary */}
            {data && (
                <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4 mb-6">
                    <div className="grid grid-cols-3 gap-4 text-center">
                        <div>
                            <div className="text-sm text-gray-600 dark:text-gray-400">Expected Points</div>
                            <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                                +{data.total_expected_points.toFixed(1)}
                            </div>
                        </div>
                        <div>
                            <div className="text-sm text-gray-600 dark:text-gray-400">Transfer Cost</div>
                            <div className="text-2xl font-bold text-red-600 dark:text-red-400">
                                -{data.total_transfer_cost}
                            </div>
                        </div>
                        <div>
                            <div className="text-sm text-gray-600 dark:text-gray-400">Net Gain</div>
                            <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                                +{(data.total_expected_points - data.total_transfer_cost).toFixed(1)}
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Gameweek Plans */}
            {isLoading ? (
                <div className="space-y-4">
                    {[1, 2, 3].map((i) => (
                        <div key={i} className="animate-pulse bg-gray-100 dark:bg-gray-700 rounded-lg h-32"></div>
                    ))}
                </div>
            ) : (
                <div className="space-y-4">
                    {data?.gameweeks.map((gw) => (
                        <div
                            key={gw.gameweek}
                            className={`border rounded-lg p-4 ${gw.should_transfer
                                    ? 'border-green-300 dark:border-green-700 bg-green-50 dark:bg-green-900/10'
                                    : 'border-gray-300 dark:border-gray-600 bg-gray-50 dark:bg-gray-700/50'
                                }`}
                        >
                            <div className="flex items-start justify-between mb-3">
                                <div>
                                    <h3 className="text-lg font-bold text-gray-900 dark:text-white">
                                        Gameweek +{gw.gameweek}
                                    </h3>
                                    <p className="text-sm text-gray-600 dark:text-gray-400">
                                        {gw.recommendation}
                                    </p>
                                </div>
                                <div className="text-right">
                                    <div className="text-xs text-gray-500 dark:text-gray-400">Free Transfers</div>
                                    <div className="text-lg font-semibold text-gray-900 dark:text-white">
                                        {gw.free_transfers_before} → {gw.free_transfers_after}
                                    </div>
                                </div>
                            </div>

                            {gw.should_transfer && gw.transfers_in.length > 0 && (
                                <div className="space-y-2">
                                    {gw.transfers_in.map((playerIn, idx) => (
                                        <div key={idx} className="flex items-center gap-3 bg-white dark:bg-gray-800 rounded-lg p-3">
                                            <div className="flex-1">
                                                <div className="flex items-center gap-2">
                                                    <span className="text-red-500">←</span>
                                                    <span className="text-sm text-gray-600 dark:text-gray-400">
                                                        {gw.transfers_out[idx]?.web_name}
                                                    </span>
                                                </div>
                                                <div className="flex items-center gap-2 mt-1">
                                                    <span className="text-green-500">→</span>
                                                    <span className="text-sm font-semibold text-gray-900 dark:text-white">
                                                        {playerIn.web_name}
                                                    </span>
                                                </div>
                                            </div>
                                            <div className="text-right">
                                                <div className="text-xs text-gray-500 dark:text-gray-400">Cost</div>
                                                <div className="text-sm font-semibold text-gray-900 dark:text-white">
                                                    £{(playerIn.now_cost / 10).toFixed(1)}m
                                                </div>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}

                            {/* Price Change Warnings */}
                            {(gw.price_change_risk.rise_soon.length > 0 || gw.price_change_risk.fall_soon.length > 0) && (
                                <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-600">
                                    {gw.price_change_risk.rise_soon.length > 0 && (
                                        <div className="text-xs text-orange-600 dark:text-orange-400">
                                            ⚠️ Price rises likely: {gw.price_change_risk.rise_soon.map(p => p.player).join(', ')}
                                        </div>
                                    )}
                                </div>
                            )}

                            {/* Stats */}
                            <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-600 flex justify-between text-sm">
                                <span className="text-gray-600 dark:text-gray-400">
                                    Expected: {gw.expected_points.toFixed(1)} pts
                                </span>
                                {gw.transfer_cost > 0 && (
                                    <span className="text-red-600 dark:text-red-400">
                                        Cost: -{gw.transfer_cost} pts
                                    </span>
                                )}
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
