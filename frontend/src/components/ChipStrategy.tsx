/**
 * Chip Strategy Component
 * Displays optimal chip usage strategy for the season
 */

'use client';

import { useQuery } from '@tanstack/react-query';
import axios from 'axios';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface ChipStrategy {
    current_gameweek: number;
    remaining_chips: string[];
    double_gameweeks: number[];
    blank_gameweeks: number[];
    recommendations: Array<{
        chip: string;
        recommended_gameweek: number | null;
        priority: number;
        reasoning: string;
        expected_benefit: number;
    }>;
    timeline: Array<{
        gameweek: number;
        chip: string;
        reasoning: string;
        priority: number;
    }>;
}

export default function ChipStrategy({ currentGameweek, usedChips }: {
    currentGameweek: number;
    usedChips: string[];
}) {
    const { data, isLoading } = useQuery<ChipStrategy>({
        queryKey: ['chip-strategy', currentGameweek, usedChips],
        queryFn: async () => {
            const params = new URLSearchParams({
                current_gameweek: currentGameweek.toString(),
                ...(usedChips.length > 0 && { used_chips: usedChips.join(',') }),
            });
            const res = await axios.get(`${API_BASE}/api/features/chip-strategy?${params}`);
            return res.data;
        },
    });

    const getChipIcon = (chip: string) => {
        const icons: Record<string, string> = {
            wildcard: '🃏',
            free_hit: '🎯',
            bench_boost: '💪',
            triple_captain: '👑',
        };
        return icons[chip] || '🎮';
    };

    const getPriorityColor = (priority: number) => {
        if (priority >= 9) return 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300';
        if (priority >= 7) return 'bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-300';
        if (priority >= 5) return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300';
        return 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300';
    };

    if (isLoading) {
        return (
            <div className="animate-pulse bg-gray-100 dark:bg-gray-800 rounded-lg p-6 h-96"></div>
        );
    }

    if (!data) return null;

    return (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
            {/* Header */}
            <div className="mb-6">
                <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
                    Chip Strategy Planner
                </h2>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                    Optimal timing for your remaining chips
                </p>
            </div>

            {/* Special Gameweeks */}
            {(data.double_gameweeks.length > 0 || data.blank_gameweeks.length > 0) && (
                <div className="grid grid-cols-2 gap-4 mb-6">
                    {data.double_gameweeks.length > 0 && (
                        <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-4">
                            <div className="text-sm font-semibold text-green-800 dark:text-green-200 mb-2">
                                Double Gameweeks
                            </div>
                            <div className="flex flex-wrap gap-2">
                                {data.double_gameweeks.map((gw) => (
                                    <span key={gw} className="px-2 py-1 bg-green-200 dark:bg-green-800 text-green-900 dark:text-green-100 rounded text-sm font-semibold">
                                        GW{gw}
                                    </span>
                                ))}
                            </div>
                        </div>
                    )}

                    {data.blank_gameweeks.length > 0 && (
                        <div className="bg-orange-50 dark:bg-orange-900/20 border border-orange-200 dark:border-orange-800 rounded-lg p-4">
                            <div className="text-sm font-semibold text-orange-800 dark:text-orange-200 mb-2">
                                Blank Gameweeks
                            </div>
                            <div className="flex flex-wrap gap-2">
                                {data.blank_gameweeks.map((gw) => (
                                    <span key={gw} className="px-2 py-1 bg-orange-200 dark:bg-orange-800 text-orange-900 dark:text-orange-100 rounded text-sm font-semibold">
                                        GW{gw}
                                    </span>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* Recommendations */}
            <div className="space-y-4 mb-6">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                    Chip Recommendations
                </h3>

                {data.recommendations.map((rec, idx) => (
                    <div
                        key={idx}
                        className="bg-gradient-to-r from-gray-50 to-white dark:from-gray-700 dark:to-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg p-4"
                    >
                        <div className="flex items-start justify-between mb-3">
                            <div className="flex items-center gap-3">
                                <span className="text-3xl">{getChipIcon(rec.chip)}</span>
                                <div>
                                    <h4 className="text-lg font-bold text-gray-900 dark:text-white capitalize">
                                        {rec.chip.replace('_', ' ')}
                                    </h4>
                                    {rec.recommended_gameweek && (
                                        <p className="text-sm text-gray-600 dark:text-gray-400">
                                            Recommended: GW{rec.recommended_gameweek}
                                        </p>
                                    )}
                                </div>
                            </div>

                            <span className={`px-3 py-1 rounded-full text-xs font-semibold ${getPriorityColor(rec.priority)}`}>
                                Priority: {rec.priority}/10
                            </span>
                        </div>

                        <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
                            {rec.reasoning}
                        </p>

                        <div className="flex items-center justify-between text-sm">
                            <span className="text-gray-600 dark:text-gray-400">
                                Expected Benefit:
                            </span>
                            <span className="font-semibold text-green-600 dark:text-green-400">
                                +{rec.expected_benefit.toFixed(1)} pts
                            </span>
                        </div>
                    </div>
                ))}

                {data.recommendations.length === 0 && (
                    <div className="text-center py-8 text-gray-500 dark:text-gray-400">
                        All chips have been used! 🎉
                    </div>
                )}
            </div>

            {/* Timeline */}
            {data.timeline.length > 0 && (
                <div>
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                        Suggested Timeline
                    </h3>

                    <div className="relative">
                        {/* Timeline line */}
                        <div className="absolute left-6 top-0 bottom-0 w-0.5 bg-gray-300 dark:bg-gray-600"></div>

                        <div className="space-y-6">
                            {data.timeline.map((item, idx) => (
                                <div key={idx} className="relative flex items-start gap-4 pl-14">
                                    {/* Timeline dot */}
                                    <div className="absolute left-4 w-4 h-4 bg-blue-500 rounded-full border-4 border-white dark:border-gray-800"></div>

                                    <div className="flex-1 bg-white dark:bg-gray-700 rounded-lg p-4 shadow-sm">
                                        <div className="flex items-center gap-2 mb-2">
                                            <span className="text-xl">{getChipIcon(item.chip)}</span>
                                            <span className="font-semibold text-gray-900 dark:text-white">
                                                GW{item.gameweek}
                                            </span>
                                            <span className="text-sm text-gray-600 dark:text-gray-400 capitalize">
                                                - {item.chip.replace('_', ' ')}
                                            </span>
                                        </div>
                                        <p className="text-sm text-gray-700 dark:text-gray-300">
                                            {item.reasoning}
                                        </p>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
