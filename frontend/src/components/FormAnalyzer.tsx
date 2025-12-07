/**
 * Form Momentum Analyzer Component
 * Displays player form analysis with trends, streaks, and breakout detection
 */

'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface FormAnalysis {
    momentum: {
        momentum_score: number;
        trend: string;
        hot_streak_length: number;
        cold_streak_length: number;
        volatility: number;
        recent_avg: number;
        season_avg: number;
    };
    breakout: {
        is_breakout: boolean;
        confidence: number;
        reason: string;
    };
    continuation: {
        continuation_probability: number;
        confidence: number;
    };
    summary: string;
}

export default function FormAnalyzer({ playerId }: { playerId: number }) {
    const { data, isLoading, error } = useQuery<FormAnalysis>({
        queryKey: ['form-analysis', playerId],
        queryFn: async () => {
            const res = await axios.get(`${API_BASE}/api/advanced/form-analysis/${playerId}`);
            return res.data;
        },
    });

    if (isLoading) {
        return (
            <div className="animate-pulse bg-gray-100 dark:bg-gray-800 rounded-lg p-6">
                <div className="h-4 bg-gray-300 dark:bg-gray-700 rounded w-3/4 mb-4"></div>
                <div className="h-4 bg-gray-300 dark:bg-gray-700 rounded w-1/2"></div>
            </div>
        );
    }

    if (error || !data) {
        return (
            <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
                <p className="text-red-800 dark:text-red-200">Failed to load form analysis</p>
            </div>
        );
    }

    const getMomentumColor = (score: number) => {
        if (score > 0.3) return 'text-green-600 dark:text-green-400';
        if (score < -0.3) return 'text-red-600 dark:text-red-400';
        return 'text-gray-600 dark:text-gray-400';
    };

    const getTrendIcon = (trend: string) => {
        if (trend === 'improving') return '📈';
        if (trend === 'declining') return '📉';
        return '➡️';
    };

    return (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 space-y-6">
            {/* Header */}
            <div className="border-b border-gray-200 dark:border-gray-700 pb-4">
                <h3 className="text-xl font-bold text-gray-900 dark:text-white">
                    Form Momentum Analysis
                </h3>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">{data.summary}</p>
            </div>

            {/* Momentum Score */}
            <div className="grid grid-cols-2 gap-4">
                <div className="bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 rounded-lg p-4">
                    <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">Momentum Score</div>
                    <div className={`text-3xl font-bold ${getMomentumColor(data.momentum.momentum_score)}`}>
                        {data.momentum.momentum_score > 0 ? '+' : ''}
                        {data.momentum.momentum_score.toFixed(2)}
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-500 mt-1">
                        {getTrendIcon(data.momentum.trend)} {data.momentum.trend}
                    </div>
                </div>

                <div className="bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/20 rounded-lg p-4">
                    <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">Form Continuation</div>
                    <div className="text-3xl font-bold text-purple-600 dark:text-purple-400">
                        {(data.continuation.continuation_probability * 100).toFixed(0)}%
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-500 mt-1">
                        Confidence: {(data.continuation.confidence * 100).toFixed(0)}%
                    </div>
                </div>
            </div>

            {/* Streaks */}
            {(data.momentum.hot_streak_length > 0 || data.momentum.cold_streak_length > 0) && (
                <div className="flex gap-3">
                    {data.momentum.hot_streak_length > 0 && (
                        <div className="flex-1 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-3">
                            <div className="flex items-center gap-2">
                                <span className="text-2xl">🔥</span>
                                <div>
                                    <div className="text-sm font-semibold text-green-800 dark:text-green-200">
                                        Hot Streak
                                    </div>
                                    <div className="text-xs text-green-600 dark:text-green-400">
                                        {data.momentum.hot_streak_length} games
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}

                    {data.momentum.cold_streak_length > 0 && (
                        <div className="flex-1 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-3">
                            <div className="flex items-center gap-2">
                                <span className="text-2xl">❄️</span>
                                <div>
                                    <div className="text-sm font-semibold text-blue-800 dark:text-blue-200">
                                        Cold Streak
                                    </div>
                                    <div className="text-xs text-blue-600 dark:text-blue-400">
                                        {data.momentum.cold_streak_length} games
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* Breakout Alert */}
            {data.breakout.is_breakout && (
                <div className="bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 border border-yellow-300 dark:border-yellow-700 rounded-lg p-4">
                    <div className="flex items-start gap-3">
                        <span className="text-3xl">⭐</span>
                        <div className="flex-1">
                            <div className="font-bold text-yellow-900 dark:text-yellow-100 mb-1">
                                Breaking Out!
                            </div>
                            <div className="text-sm text-yellow-800 dark:text-yellow-200">
                                {data.breakout.reason}
                            </div>
                            <div className="text-xs text-yellow-700 dark:text-yellow-300 mt-1">
                                Confidence: {(data.breakout.confidence * 100).toFixed(0)}%
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Stats Grid */}
            <div className="grid grid-cols-3 gap-3 pt-4 border-t border-gray-200 dark:border-gray-700">
                <div className="text-center">
                    <div className="text-xs text-gray-500 dark:text-gray-400">Recent Avg</div>
                    <div className="text-lg font-semibold text-gray-900 dark:text-white">
                        {data.momentum.recent_avg.toFixed(1)}
                    </div>
                </div>
                <div className="text-center">
                    <div className="text-xs text-gray-500 dark:text-gray-400">Season Avg</div>
                    <div className="text-lg font-semibold text-gray-900 dark:text-white">
                        {data.momentum.season_avg.toFixed(1)}
                    </div>
                </div>
                <div className="text-center">
                    <div className="text-xs text-gray-500 dark:text-gray-400">Volatility</div>
                    <div className="text-lg font-semibold text-gray-900 dark:text-white">
                        {data.momentum.volatility.toFixed(1)}
                    </div>
                </div>
            </div>
        </div>
    );
}
