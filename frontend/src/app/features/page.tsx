/**
 * Advanced Features Page
 * Main page showcasing all new features
 */

'use client';

import { useState } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import FormAnalyzer from '@/components/FormAnalyzer';
import DifferentialFinder from '@/components/DifferentialFinder';
import MultiGWPlanner from '@/components/MultiGWPlanner';
import ChipStrategy from '@/components/ChipStrategy';
import InjuryRiskDashboard from '@/components/InjuryRiskDashboard';

const queryClient = new QueryClient({
    defaultOptions: {
        queries: {
            refetchOnWindowFocus: false,
            retry: 1,
        },
    },
});

export default function AdvancedFeaturesPage() {
    const [activeTab, setActiveTab] = useState('differentials');
    const [selectedPlayer, setSelectedPlayer] = useState<number | null>(null);
    const [currentTeam, setCurrentTeam] = useState<number[]>([]);
    const [currentGameweek, setCurrentGameweek] = useState(10);
    const [usedChips, setUsedChips] = useState<string[]>([]);

    const tabs = [
        { id: 'differentials', name: 'Differentials', icon: '💎' },
        { id: 'form', name: 'Form Analysis', icon: '📊' },
        { id: 'transfers', name: 'Transfer Planner', icon: '🔄' },
        { id: 'chips', name: 'Chip Strategy', icon: '🎮' },
        { id: 'injury', name: 'Injury Risk', icon: '🏥' },
    ];

    return (
        <QueryClientProvider client={queryClient}>
            <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
                {/* Header */}
                <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
                    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
                        <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
                            Advanced FPL Features
                        </h1>
                        <p className="text-gray-600 dark:text-gray-400">
                            AI-powered analysis and strategic planning tools
                        </p>
                    </div>
                </div>

                {/* Tabs */}
                <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
                    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                        <nav className="flex space-x-8" aria-label="Tabs">
                            {tabs.map((tab) => (
                                <button
                                    key={tab.id}
                                    onClick={() => setActiveTab(tab.id)}
                                    className={`
                    py-4 px-1 border-b-2 font-medium text-sm flex items-center gap-2
                    ${activeTab === tab.id
                                            ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                                            : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 dark:text-gray-400 dark:hover:text-gray-300'
                                        }
                  `}
                                >
                                    <span>{tab.icon}</span>
                                    <span>{tab.name}</span>
                                </button>
                            ))}
                        </nav>
                    </div>
                </div>

                {/* Content */}
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                    {activeTab === 'differentials' && <DifferentialFinder />}

                    {activeTab === 'form' && (
                        <div className="space-y-6">
                            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                                    Select Player ID for Analysis
                                </label>
                                <input
                                    type="number"
                                    value={selectedPlayer || ''}
                                    onChange={(e) => setSelectedPlayer(Number(e.target.value))}
                                    placeholder="Enter player ID"
                                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                                />
                            </div>
                            {selectedPlayer && <FormAnalyzer playerId={selectedPlayer} />}
                        </div>
                    )}

                    {activeTab === 'transfers' && (
                        <div className="space-y-6">
                            <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-4">
                                <p className="text-sm text-yellow-800 dark:text-yellow-200">
                                    💡 <strong>Note:</strong> Connect your FPL team to use the transfer planner.
                                    For demo purposes, you can manually enter 15 player IDs.
                                </p>
                            </div>
                            <MultiGWPlanner currentTeam={currentTeam} />
                        </div>
                    )}

                    {activeTab === 'chips' && (
                        <div className="space-y-6">
                            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                                <div className="grid grid-cols-2 gap-4">
                                    <div>
                                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                                            Current Gameweek
                                        </label>
                                        <input
                                            type="number"
                                            min="1"
                                            max="38"
                                            value={currentGameweek}
                                            onChange={(e) => setCurrentGameweek(Number(e.target.value))}
                                            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700"
                                        />
                                    </div>
                                    <div>
                                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                                            Used Chips
                                        </label>
                                        <select
                                            multiple
                                            value={usedChips}
                                            onChange={(e) => setUsedChips(Array.from(e.target.selectedOptions, option => option.value))}
                                            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700"
                                        >
                                            <option value="wildcard">Wildcard</option>
                                            <option value="free_hit">Free Hit</option>
                                            <option value="bench_boost">Bench Boost</option>
                                            <option value="triple_captain">Triple Captain</option>
                                        </select>
                                    </div>
                                </div>
                            </div>
                            <ChipStrategy currentGameweek={currentGameweek} usedChips={usedChips} />
                        </div>
                    )}

                    {activeTab === 'injury' && <InjuryRiskDashboard />}
                </div>

                {/* Footer */}
                <div className="bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 mt-12">
                    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
                        <div className="text-center text-sm text-gray-600 dark:text-gray-400">
                            <p>
                                Powered by AI • Data refreshes hourly •
                                <a href="/api/docs" className="text-blue-600 dark:text-blue-400 hover:underline ml-1">
                                    API Documentation
                                </a>
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </QueryClientProvider>
    );
}
