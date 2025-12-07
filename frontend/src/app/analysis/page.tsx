'use client';

import { useEffect, useState, useMemo, useCallback } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import { Player, AdvancedOptimizationResult, ManagerPreferences, Club } from '@/lib/types';
import { formatPrice, getPositionIcon, getTeamColor } from '@/lib/utils';
import { FPLApi } from '@/lib/api';
import { getAnalysisSession } from '@/lib/analysisSession';
import FormAnalyzer from '@/components/FormAnalyzer';
import DifferentialFinder from '@/components/DifferentialFinder';
import MultiGWPlanner from '@/components/MultiGWPlanner';
import ChipStrategy from '@/components/ChipStrategy';
import InjuryRiskDashboard from '@/components/InjuryRiskDashboard';

interface TeamConfig {
  free_transfers: number;
  bank: number;
  chips_available: string[];
  starting_xi: number[];
}

const DEFAULT_TEAM_CONFIG: TeamConfig = {
  free_transfers: 1,
  bank: 0.0,
  chips_available: ['Wildcard', 'Bench Boost', 'Triple Captain', 'Free Hit'],
  starting_xi: []
};

const deriveFormation = (team: Player[], startingXI: number[] = []) => {
  const xi = Array.isArray(startingXI) ? startingXI : [];
  if (!xi.length || !team.length) {
    return '3-4-3';
  }
  const counts = { GKP: 0, DEF: 0, MID: 0, FWD: 0 };
  const playerMap = new Map<number, Player>();
  team.forEach((player) => {
    playerMap.set(player.id, player);
  });
  xi.slice(0, 11).forEach((playerId) => {
    const player = playerMap.get(playerId);
    if (!player) {
      return;
    }
    const position = player.position as keyof typeof counts;
    if (counts[position] !== undefined) {
      counts[position] += 1;
    }
  });
  if (counts.DEF === 0 && counts.MID === 0 && counts.FWD === 0) {
    return '3-4-3';
  }
  return `${counts.DEF}-${counts.MID}-${counts.FWD}`;
};

export default function AdvancedAnalysisPage() {
  const [team, setTeam] = useState<Player[]>([]);
  const [analysisData, setAnalysisData] = useState<AdvancedOptimizationResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [preferences, setPreferences] = useState<ManagerPreferences>({
    risk_tolerance: 0.5,
    formation_preference: '3-4-3',
    budget_allocation: { GKP: 0.08, DEF: 0.25, MID: 0.40, FWD: 0.27 },
    differential_threshold: 5.0,
    captain_strategy: 'fixture_based',
    transfer_frequency: 'moderate'
  });
  const [analysisOptions, setAnalysisOptions] = useState({
    include_fixture_analysis: true,
    include_price_analysis: true,
    include_strategic_planning: true
  });
  const [teamsReference, setTeamsReference] = useState<Club[]>([]);
  const [teamConfig, setTeamConfig] = useState<TeamConfig>(() => ({ ...DEFAULT_TEAM_CONFIG }));
  const [activeTab, setActiveTab] = useState<'overview' | 'transfers' | 'chips' | 'bench' | 'advanced' | 'form' | 'differentials' | 'injury'>('overview');

  const searchParams = useSearchParams();
  const router = useRouter();

  useEffect(() => {
    let mounted = true;
    const loadTeams = async () => {
      try {
        const data = await FPLApi.getTeams();
        if (mounted && Array.isArray(data)) {
          setTeamsReference(data);
        }
      } catch (err) {
        console.error('Failed to fetch teams reference', err);
      }
    };
    loadTeams();
    return () => {
      mounted = false;
    };
  }, []);

  useEffect(() => {
    const teamData = searchParams.get('team');
    const configData = searchParams.get('config');
    const sessionId = searchParams.get('session');
    const loadFromSession = () => {
      if (!sessionId) return false;
      const payload = getAnalysisSession(sessionId);
      if (!payload) {
        setError('Your analysis session expired. Please upload your team again.');
        setLoading(false);
        return true;
      }
      setTeam(payload.players);
      const effectiveConfig = {
        ...DEFAULT_TEAM_CONFIG,
        ...payload.config,
        chips_available: payload.config?.chips_available ?? DEFAULT_TEAM_CONFIG.chips_available,
        starting_xi: Array.isArray(payload.config?.starting_xi) ? payload.config.starting_xi : []
      };
      setTeamConfig(effectiveConfig);
      fetchAdvancedAnalysis(payload.players, effectiveConfig);
      return true;
    };

    if (loadFromSession()) {
      return;
    }

    if (!teamData) {
      router.push('/');
      return;
    }

    try {
      // Safely decode URI component - try decoding first, fallback to direct parse
      let decodedTeamData: string;
      try {
        decodedTeamData = decodeURIComponent(teamData);
      } catch (decodeError) {
        // If decode fails, the data might already be decoded or malformed
        // Try parsing directly
        decodedTeamData = teamData;
      }

      const parsedTeam = JSON.parse(decodedTeamData);

      // Validate parsed team is an array
      if (!Array.isArray(parsedTeam)) {
        throw new Error('Team data must be an array');
      }

      setTeam(parsedTeam);

      // Parse config if available
      let effectiveConfig: TeamConfig = { ...DEFAULT_TEAM_CONFIG };
      if (configData) {
        try {
          let decodedConfigData: string;
          try {
            decodedConfigData = decodeURIComponent(configData);
          } catch (decodeError) {
            decodedConfigData = configData;
          }

          const parsedConfig = JSON.parse(decodedConfigData);
          effectiveConfig = {
            free_transfers: parsedConfig.free_transfers ?? 1,
            bank: parsedConfig.bank ?? parsedConfig.budget ?? 0.0,
            chips_available: parsedConfig.chips_available ?? ['Wildcard', 'Bench Boost', 'Triple Captain', 'Free Hit'],
            starting_xi: Array.isArray(parsedConfig.starting_xi) ? parsedConfig.starting_xi : []
          };
        } catch (e) {
          console.error('Error parsing config:', e);
        }
      }
      setTeamConfig(effectiveConfig);

      // Fetch advanced analysis data with the parsed config
      fetchAdvancedAnalysis(parsedTeam, effectiveConfig);
    } catch (error) {
      console.error('Error parsing team data:', error);
      setError('Invalid team data. Please upload your team again.');
      setLoading(false);
    }
  }, [searchParams, router]);

  const calculatedFormation = deriveFormation(team, teamConfig.starting_xi);
  const calculateTeamValue = (players: Player[]) =>
    players.reduce((sum, player) => sum + (player?.now_cost || 0) / 10, 0);

  const fetchAdvancedAnalysis = async (teamPlayers: Player[], configOverride?: TeamConfig) => {
    try {
      setLoading(true);
      const effectiveConfig = configOverride ?? teamConfig;
      const totalTeamValue = calculateTeamValue(teamPlayers);
      const totalBudget = Number((totalTeamValue + (effectiveConfig.bank ?? 0)).toFixed(1));
      const result = await FPLApi.advancedOptimizeTeam({
        players: teamPlayers.map(p => p.id),
        budget: totalBudget,
        bank_amount: Number(effectiveConfig.bank ?? 0),
        free_transfers: effectiveConfig.free_transfers,
        use_wildcard: false,
        chips_available: effectiveConfig.chips_available,
        starting_xi: effectiveConfig.starting_xi,
        formation: calculatedFormation || '3-4-3',
        // preferences: preferences,  // Temporarily commented out to test
        ...analysisOptions
      });

      setAnalysisData(result);
    } catch (error) {
      console.error('Advanced analysis failed:', error);
      setError('Failed to analyze team. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleReanalyze = () => {
    fetchAdvancedAnalysis(team, teamConfig);
  };

  const getConfidenceColor = (score: number) => {
    if (score >= 0.8) return 'text-green-600 bg-green-100';
    if (score >= 0.6) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'High': return 'bg-red-100 text-red-800 border-red-200';
      case 'Medium': return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'Low': return 'bg-green-100 text-green-800 border-green-200';
      default: return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const benchData = analysisData?.bench_analysis || null;
  const transferSuggestions = analysisData?.transfer_suggestions ?? [];
  const allChips = ['Wildcard', 'Bench Boost', 'Triple Captain', 'Free Hit'];
  const availableChips = analysisData?.team_analysis?.chips_available ?? teamConfig.chips_available;
  const chipsAvailableSet = new Set(availableChips);
  const chipOpportunities = (analysisData?.chip_opportunities || []).filter(
    (chip: any) => chipsAvailableSet.has(chip.chip)
  );
  const predictedPointsValue = analysisData?.team_analysis?.predicted_points;
  const predictedPointsDisplay = typeof predictedPointsValue === 'number' ? predictedPointsValue.toFixed(1) : '0.0';
  const teamStrengthValue = analysisData?.team_analysis?.team_strength;
  const teamStrengthDisplay = typeof teamStrengthValue === 'number' ? teamStrengthValue.toFixed(1) : '0.0';
  const formationDisplay = analysisData?.team_analysis?.formation || calculatedFormation || '3-4-3';
  const freeTransfersDisplay = analysisData?.team_analysis?.free_transfers ?? teamConfig.free_transfers;
  const freeTransfersCount = typeof freeTransfersDisplay === 'number' ? freeTransfersDisplay : Number(freeTransfersDisplay ?? 0);
  const hasFreeTransfers = freeTransfersCount > 0;
  const bankValue =
    typeof analysisData?.team_analysis?.bank_remaining === 'number'
      ? analysisData.team_analysis.bank_remaining
      : teamConfig.bank ?? 0;
  const bankDisplay = bankValue.toFixed(1);
  const rawAutosub = Array.isArray(benchData?.autosub_potential) ? benchData.autosub_potential : [];
  const autosubRecommendations = useMemo(() => {
    const seen = new Set<string>();
    return rawAutosub.filter((entry: any) => {
      const benchId = entry?.bench_player?.id ?? entry?.bench_player?.web_name;
      const starterId = entry?.starter_to_replace?.id ?? entry?.starter_to_replace?.web_name;
      if (!benchId || !starterId) {
        return false;
      }
      const key = `${benchId}-${starterId}`;
      if (seen.has(key)) {
        return false;
      }
      seen.add(key);
      return true;
    });
  }, [rawAutosub]);
  const doubleGameweeksRaw = analysisData?.fixture_analysis?.double_gameweeks;
  const blankGameweeksRaw = analysisData?.fixture_analysis?.blank_gameweeks;
  const doubleGameweeks = Array.isArray(doubleGameweeksRaw) ? doubleGameweeksRaw : [];
  const blankGameweeks = Array.isArray(blankGameweeksRaw) ? blankGameweeksRaw : [];
  const fixtureDoubleCount = doubleGameweeks.length;
  const fixtureBlankCount = blankGameweeks.length;
  const risingPlayers = analysisData?.price_analysis?.rising_players ?? [];
  const fallingPlayers = analysisData?.price_analysis?.falling_players ?? [];
  const risingCount = risingPlayers.length;
  const fallingCount = fallingPlayers.length;
  const newsRisks = analysisData?.news_analysis?.high_risk_players ?? [];
  const newsRiskCount = newsRisks.length;
  const differentialCount = analysisData?.effective_ownership?.differential_opportunities?.length ?? 0;
  const teamNameToId = useMemo(() => {
    const map = new Map<string, number>();
    teamsReference.forEach((team) => {
      map.set(team.name.toLowerCase(), team.id);
    });
    return map;
  }, [teamsReference]);
  const teamIdToName = useMemo(() => {
    const map = new Map<number, string>();
    teamsReference.forEach((team) => {
      map.set(team.id, team.name);
    });
    return map;
  }, [teamsReference]);
  const getFixtureMeta = useCallback(
    (teamName?: string) => {
      if (!teamName || !analysisData?.fixture_analysis?.team_summaries) {
        return null;
      }
      const teamId = teamNameToId.get(teamName.toLowerCase());
      if (!teamId) {
        return null;
      }
      const summaries = analysisData.fixture_analysis.team_summaries;
      const summary = summaries[teamId] || summaries[String(teamId)];
      if (!summary || !summary.next_fixture) {
        return null;
      }
      const opponentId = summary.next_fixture.opponent_team;
      return {
        teamId,
        opponentId,
        opponentName: (opponentId && teamIdToName.get(opponentId)) || 'TBD',
        difficulty: summary.next_fixture.difficulty,
        venue: summary.next_fixture.venue
      };
    },
    [analysisData?.fixture_analysis?.team_summaries, teamNameToId, teamIdToName]
  );
  const captainCards = useMemo(() => {
    const suggestions = analysisData?.captain_suggestions ?? [];
    return suggestions
      .map((suggestion) => {
        const meta = getFixtureMeta(suggestion.player.team_name);
        const difficulty = meta?.difficulty ?? null;
        let difficultyWeight = 1;
        if (typeof difficulty === 'number') {
          if (difficulty <= 2) {
            difficultyWeight = 1.1;
          } else if (difficulty >= 4) {
            difficultyWeight = 0.85;
          }
        }
        const adjustedScore = (suggestion.expected_points ?? 0) * difficultyWeight;
        return {
          suggestion,
          meta,
          adjustedScore
        };
      })
      .sort((a, b) => b.adjustedScore - a.adjustedScore);
  }, [analysisData?.captain_suggestions, getFixtureMeta]);
  const orderedTransferSuggestions = useMemo(() => {
    if (!transferSuggestions?.length) {
      return [];
    }
    const rank: Record<string, number> = { High: 0, Medium: 1, Low: 2 };
    return [...transferSuggestions].sort((a, b) => {
      const aRank = rank[a.priority] ?? 3;
      const bRank = rank[b.priority] ?? 3;
      if (aRank !== bRank) {
        return aRank - bRank;
      }
      return (b.points_gain ?? 0) - (a.points_gain ?? 0);
    });
  }, [transferSuggestions]);
  const transferCardElements = useMemo(() => {
    return orderedTransferSuggestions.map((transfer, index) => {
      const outgoingPlayer = transfer.player_out ?? ({} as Player);
      const incomingPlayer = transfer.player_in ?? ({} as Player);
      const incomingFixture = getFixtureMeta(incomingPlayer.team_name);
      const metrics = transfer.metrics ?? {};
      const startProbability =
        typeof metrics.start_probability === 'number' ? `${(metrics.start_probability * 100).toFixed(0)}%` : '—';
      const expectedMinutes =
        typeof metrics.expected_minutes === 'number' ? `${metrics.expected_minutes.toFixed(0)} mins` : '—';
      const gaPer90 = typeof metrics.ga_per90 === 'number' ? metrics.ga_per90.toFixed(2) : '—';
      const xgiPer90 = typeof metrics.xgi_per90 === 'number' ? metrics.xgi_per90.toFixed(2) : '—';
      const assistsPer90 = typeof metrics.assists_per90 === 'number' ? metrics.assists_per90.toFixed(2) : '—';
      const tacklesPer90 = typeof metrics.tackles_per90 === 'number' ? metrics.tackles_per90.toFixed(1) : '—';
      const defContribPer90 = typeof metrics.defensive_contrib_per90 === 'number' ? metrics.defensive_contrib_per90.toFixed(1) : '—';
      const cbiPer90 = typeof metrics.cbi_per90 === 'number' ? metrics.cbi_per90.toFixed(1) : '—';
      const bpsPer90 = typeof metrics.bps_per90 === 'number' ? metrics.bps_per90.toFixed(1) : '—';
      const impliedConceded = typeof metrics.implied_conceded === 'number' ? metrics.implied_conceded.toFixed(2) : '—';
      const playerPosition = metrics.position || incomingPlayer.position || 'MID';
      const fixtureDifficulty =
        typeof metrics.fixture_difficulty === 'number' ? metrics.fixture_difficulty : incomingFixture?.difficulty;
      const fixtureDelta = typeof metrics.fixture_delta === 'number' ? metrics.fixture_delta.toFixed(2) : null;
      const impliedGoals =
        typeof metrics.implied_goals === 'number' ? metrics.implied_goals.toFixed(2) : undefined;
      const ownershipDelta =
        typeof metrics.ownership_delta === 'number'
          ? `${metrics.ownership_delta >= 0 ? '+' : ''}${metrics.ownership_delta.toFixed(1)}%`
          : '—';
      const incomingFixtureLabel = incomingFixture
        ? `${incomingFixture.venue === 'home' ? 'vs' : '@'} ${incomingFixture.opponentName}`
        : 'Fixture TBC';
      const showFreeBadge = Boolean(transfer.is_free);
      const cardClass = showFreeBadge
        ? 'bg-gradient-to-r from-green-50 to-emerald-50 border-green-300'
        : 'bg-gradient-to-r from-orange-50 to-red-50 border-orange-300';
      const transferCost = typeof transfer.transfer_cost === 'number' ? transfer.transfer_cost : null;
      const budgetImpact =
        transfer.cost_change === 0
          ? 'Neutral'
          : transfer.cost_change > 0
            ? `+£${transfer.cost_change.toFixed(1)}m spend`
            : `Frees £${Math.abs(transfer.cost_change).toFixed(1)}m`;
      const bankRemainder =
        typeof transfer.bank_remaining === 'number' ? `Bank after: £${transfer.bank_remaining.toFixed(1)}m` : null;

      return (
        <div
          key={`${outgoingPlayer.id ?? 'out'}-${incomingPlayer.id ?? 'in'}-${index}`}
          className={`p-4 rounded-xl border-2 ${cardClass}`}
        >
          <div className="flex items-center justify-between mb-4 flex-wrap gap-3">
            <div className="flex items-center flex-wrap gap-2">
              <div className={`text-xs px-3 py-1 rounded-full font-medium border ${getPriorityColor(transfer.priority)}`}>
                {transfer.priority} Priority
              </div>
              {showFreeBadge ? (
                <div className="text-xs px-3 py-1 rounded-full font-medium bg-green-100 text-green-800 border border-green-200">
                  Free transfer
                </div>
              ) : transferCost !== null ? (
                <div className="text-xs px-3 py-1 rounded-full font-medium bg-red-100 text-red-800 border border-red-200">
                  Cost {transferCost} pts
                </div>
              ) : transfer.cost_warning ? (
                <div className="text-xs px-3 py-1 rounded-full font-medium bg-red-100 text-red-800 border border-red-200">
                  {transfer.cost_warning}
                </div>
              ) : (
                <div className="text-xs px-3 py-1 rounded-full font-medium bg-red-100 text-red-800 border border-red-200">
                  Cost pending
                </div>
              )}
              <div className="text-xs px-3 py-1 rounded-full font-medium bg-blue-100 text-blue-800 border border-blue-200">
                Net gain +{transfer.points_gain.toFixed(1)} pts
              </div>
            </div>
            <div className="text-right text-xs text-gray-500">
              {bankRemainder ?? ''}
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-center">
            <div className="text-center">
              <div className="text-xs text-red-600 font-medium mb-2">
                Transfer out
              </div>
              <div className="p-4 bg-white rounded-xl shadow-sm border-2 border-red-200">
                <div className="mb-3">
                  <div className="w-16 h-16 mx-auto bg-gradient-to-br from-gray-100 to-gray-200 rounded-full flex items-center justify-center text-2xl">
                    {getPositionIcon(outgoingPlayer.position)}
                  </div>
                </div>
                <div className="font-bold text-gray-900 text-sm mb-1">{outgoingPlayer.web_name || 'Unknown player'}</div>
                <div className="text-xs text-gray-600 mb-2">{outgoingPlayer.team_name || '—'}</div>
              </div>
            </div>

            <div className="text-center flex flex-col items-center justify-center text-blue-600 text-sm font-semibold">
              Swap
            </div>

            <div className="text-center">
              <div className="text-xs text-green-600 font-medium mb-2">
                Transfer in
              </div>
              <div className="p-4 bg-white rounded-xl shadow-sm border-2 border-green-200">
                <div className="mb-3">
                  <div className="w-16 h-16 mx-auto bg-gradient-to-br from-green-100 to-emerald-200 rounded-full flex items-center justify-center text-2xl">
                    {getPositionIcon(incomingPlayer.position)}
                  </div>
                </div>
                <div className="font-bold text-gray-900 text-sm mb-1">{incomingPlayer.web_name || 'Unknown player'}</div>
                <div className="text-xs text-gray-600 mb-2">{incomingPlayer.team_name || '—'}</div>
              </div>
            </div>
          </div>

          <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-3 text-xs text-gray-700">
            <div>
              <div className="font-semibold text-gray-900">Start odds</div>
              <div>{startProbability}</div>
              <div className="text-[11px] text-gray-500">{expectedMinutes}</div>
            </div>
            <div>
              <div className="font-semibold text-gray-900">Per-90 metrics</div>
              {playerPosition === 'FWD' && (
                <>
                  <div>G+A {gaPer90}</div>
                  <div className="text-[11px] text-gray-500">xGI {xgiPer90}</div>
                </>
              )}
              {playerPosition === 'MID' && (
                <>
                  {typeof metrics.ga_per90 === 'number' && metrics.ga_per90 >= 0.35 ? (
                    <>
                      <div>G+A {gaPer90}</div>
                      <div className="text-[11px] text-gray-500">Assists {assistsPer90}</div>
                    </>
                  ) : typeof metrics.tackles_per90 === 'number' && metrics.tackles_per90 >= 2.5 ? (
                    <>
                      <div>Tackles {tacklesPer90}</div>
                      <div className="text-[11px] text-gray-500">Def contrib {defContribPer90} · BPS {bpsPer90}</div>
                    </>
                  ) : (
                    <>
                      <div>Assists {assistsPer90}</div>
                      <div className="text-[11px] text-gray-500">xGI {xgiPer90}</div>
                    </>
                  )}
                </>
              )}
              {playerPosition === 'DEF' && (
                <>
                  {typeof metrics.ga_per90 === 'number' && metrics.ga_per90 >= 0.15 ? (
                    <>
                      <div>G+A {gaPer90}</div>
                      <div className="text-[11px] text-gray-500">Attacking defender</div>
                    </>
                  ) : (
                    <>
                      <div>Tackles {tacklesPer90}</div>
                      <div className="text-[11px] text-gray-500">CBI {cbiPer90}</div>
                    </>
                  )}
                </>
              )}
              {playerPosition === 'GKP' && (
                <>
                  <div>CS odds</div>
                  <div className="text-[11px] text-gray-500">Implied conceded {impliedConceded}</div>
                </>
              )}
            </div>
            <div>
              <div className="font-semibold text-gray-900">Fixture</div>
              <div>{incomingFixtureLabel}</div>
              <div className="text-[11px] text-gray-500">
                {typeof fixtureDifficulty === 'number' ? `Diff ${fixtureDifficulty}/5` : 'Difficulty TBD'}
                {fixtureDelta && ` · Δ ${fixtureDelta}`}
                {impliedGoals && ` · IG ${impliedGoals}`}
              </div>
            </div>
            <div>
              <div className="font-semibold text-gray-900">Budget & ownership</div>
              <div>{budgetImpact}</div>
              <div className="text-[11px] text-gray-500">Ownership Δ {ownershipDelta}</div>
            </div>
          </div>

          <div className="mt-4 space-y-3">
            {transfer.bench_guidance && (
              <div
                className={`text-sm p-3 rounded-lg border ${transfer.should_start ? 'bg-green-50 border-green-200' : 'bg-gray-50 border-gray-200'
                  }`}
              >
                <div className="flex items-center space-x-2">
                  <span className={`text-xs font-semibold uppercase ${transfer.should_start ? 'text-green-600' : 'text-gray-600'}`}>
                    {transfer.should_start ? 'Start' : 'Bench'}
                  </span>
                  <div>
                    <strong className="text-gray-900">Lineup:</strong>
                    <span className={`ml-2 ${transfer.should_start ? 'text-green-700 font-medium' : 'text-gray-700'}`}>
                      {transfer.bench_guidance}
                    </span>
                  </div>
                </div>
              </div>
            )}

            <div className="text-sm text-gray-700 bg-white/90 p-3 rounded-lg border border-gray-200">
              <div className="flex items-start space-x-2">
                <span className="text-blue-600 flex-shrink-0 text-xs font-semibold uppercase">Note</span>
                <div>
                  <strong className="text-gray-900">Reason:</strong>
                  <span className="ml-2">{transfer.reason}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      );
    });
  }, [orderedTransferSuggestions, getFixtureMeta]);

  if (loading) {
    return <AnalysisLoading />;
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 flex items-center justify-center">
        <div className="text-center">
          <div className="text-6xl mb-4 text-red-500">X</div>
          <h2 className="text-2xl font-bold text-gray-900 mb-2">Analysis Failed</h2>
          <p className="text-gray-600 mb-4">{error}</p>
          <button
            onClick={() => router.push('/')}
            className="px-6 py-3 bg-blue-600 text-white rounded-xl hover:bg-blue-700 transition-colors"
          >
            Back to team selection
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50">
      {/* Enhanced Header */}
      <div className="bg-white/80 backdrop-blur-sm border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <button
                onClick={() => router.push('/')}
                className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
              >
                Back
              </button>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Advanced FPL analysis</h1>
                <p className="text-gray-600 text-sm">AI-powered comprehensive team optimization</p>
              </div>
            </div>

            {/* Confidence Score */}
            {analysisData ? (
              <div className="text-right">
                <div className="text-sm text-gray-600">AI Confidence</div>
                <div className={`text-xl font-bold px-3 py-1 rounded-full ${getConfidenceColor(analysisData.confidence_score)}`}>
                  {(analysisData.confidence_score * 100).toFixed(0)}%
                </div>
                <div className="text-xs text-gray-500">
                  {analysisData.features_used?.total_features || 0}/6 features active
                </div>
              </div>
            ) : (
              <div className="text-right text-sm text-gray-500">
                AI Confidence pending...
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">

          {/* Sidebar - Team Overview & Settings */}
          <div className="lg:col-span-1 space-y-6">

            {/* Team Overview */}
            <div className="bg-white rounded-2xl shadow-lg p-6 border border-gray-200">
              <h3 className="text-xl font-bold text-gray-900 mb-4">
                Team overview
              </h3>

              <div className="space-y-4">
                <div className="flex justify-between items-center p-3 bg-blue-50 rounded-lg">
                  <span className="text-gray-700">Formation</span>
                  <span className="font-bold text-blue-600">{formationDisplay}</span>
                </div>

                <div className="flex justify-between items-center p-3 bg-green-50 rounded-lg">
                  <span className="text-gray-700">Predicted Points</span>
                  <span className="font-bold text-green-600">
                    {predictedPointsDisplay} pts
                  </span>
                </div>

                <div className="flex justify-between items-center p-3 bg-purple-50 rounded-lg">
                  <span className="text-gray-700">Team Strength</span>
                  <span className="font-bold text-purple-600">
                    {teamStrengthDisplay}/10
                  </span>
                </div>

                <div className="flex justify-between items-center p-3 bg-yellow-50 rounded-lg">
                  <span className="text-gray-700">Free Transfers</span>
                  <span className="font-bold text-yellow-600">{freeTransfersDisplay}</span>
                </div>

                <div className="flex justify-between items-center p-3 bg-indigo-50 rounded-lg">
                  <span className="text-gray-700">Bank</span>
                  <span className="font-bold text-indigo-600">
                    £{bankDisplay}m
                  </span>
                </div>
              </div>

              {/* Chips Status */}
              <div className="mt-6">
                <h4 className="font-semibold text-gray-900 mb-3">Chip status</h4>
                <div className="grid grid-cols-2 gap-2">
                  {allChips.map((chip) => {
                    const isAvailable = chipsAvailableSet.has(chip);
                    return (
                      <div
                        key={chip}
                        className={`p-2 rounded-lg text-xs font-medium text-center ${isAvailable ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-500'
                          }`}
                      >
                        {chip} - {isAvailable ? 'Available' : 'Used'}
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Analysis Settings */}
            <div className="bg-white rounded-2xl shadow-lg p-6 border border-gray-200">
              <h3 className="text-xl font-bold text-gray-900 mb-4">Analysis settings</h3>

              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-700">Fixture Analysis</span>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={analysisOptions.include_fixture_analysis}
                      onChange={(e) => setAnalysisOptions(prev => ({
                        ...prev,
                        include_fixture_analysis: e.target.checked
                      }))}
                      className="sr-only peer"
                    />
                    <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                  </label>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-700">Price Analysis</span>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={analysisOptions.include_price_analysis}
                      onChange={(e) => setAnalysisOptions(prev => ({
                        ...prev,
                        include_price_analysis: e.target.checked
                      }))}
                      className="sr-only peer"
                    />
                    <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                  </label>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-700">Strategic Planning</span>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={analysisOptions.include_strategic_planning}
                      onChange={(e) => setAnalysisOptions(prev => ({
                        ...prev,
                        include_strategic_planning: e.target.checked
                      }))}
                      className="sr-only peer"
                    />
                    <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                  </label>
                </div>
              </div>

              <button
                onClick={() => fetchAdvancedAnalysis(team)}
                className="w-full mt-4 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                Refresh analysis
              </button>
            </div>
          </div>

          {/* Main Content */}
          <div className="lg:col-span-3 space-y-6">

            {/* Tab Navigation */}
            <div className="bg-white rounded-2xl shadow-lg border border-gray-200 overflow-hidden">
              <div className="flex border-b border-gray-200 overflow-x-auto">
                <button
                  onClick={() => setActiveTab('overview')}
                  className={`px-6 py-4 font-medium text-sm whitespace-nowrap transition-colors ${activeTab === 'overview'
                    ? 'bg-blue-50 text-blue-600 border-b-2 border-blue-600'
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                    }`}
                >
                  Overview
                </button>
                <button
                  onClick={() => setActiveTab('transfers')}
                  className={`px-6 py-4 font-medium text-sm whitespace-nowrap transition-colors ${activeTab === 'transfers'
                    ? 'bg-blue-50 text-blue-600 border-b-2 border-blue-600'
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                    }`}
                >
                  Transfers {transferSuggestions.length > 0 && `(${transferSuggestions.length})`}
                </button>
                <button
                  onClick={() => setActiveTab('chips')}
                  className={`px-6 py-4 font-medium text-sm whitespace-nowrap transition-colors ${activeTab === 'chips'
                    ? 'bg-blue-50 text-blue-600 border-b-2 border-blue-600'
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                    }`}
                >
                  Captain & chips
                </button>
                <button
                  onClick={() => setActiveTab('bench')}
                  className={`px-6 py-4 font-medium text-sm whitespace-nowrap transition-colors ${activeTab === 'bench'
                    ? 'bg-blue-50 text-blue-600 border-b-2 border-blue-600'
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                    }`}
                >
                  Bench & formation
                </button>
                <button
                  onClick={() => setActiveTab('advanced')}
                  className={`px-6 py-4 font-medium text-sm whitespace-nowrap transition-colors ${activeTab === 'advanced'
                    ? 'bg-blue-50 text-blue-600 border-b-2 border-blue-600'
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                    }`}
                >
                  Advanced
                </button>
                <button
                  onClick={() => setActiveTab('form')}
                  className={`px-6 py-4 font-medium text-sm whitespace-nowrap transition-colors ${activeTab === 'form'
                    ? 'bg-blue-50 text-blue-600 border-b-2 border-blue-600'
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                    }`}
                >
                  📊 Form Analysis
                </button>
                <button
                  onClick={() => setActiveTab('differentials')}
                  className={`px-6 py-4 font-medium text-sm whitespace-nowrap transition-colors ${activeTab === 'differentials'
                    ? 'bg-blue-50 text-blue-600 border-b-2 border-blue-600'
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                    }`}
                >
                  💎 Differentials
                </button>
                <button
                  onClick={() => setActiveTab('injury')}
                  className={`px-6 py-4 font-medium text-sm whitespace-nowrap transition-colors ${activeTab === 'injury'
                    ? 'bg-blue-50 text-blue-600 border-b-2 border-blue-600'
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                    }`}
                >
                  🏥 Injury Risk
                </button>
              </div>
            </div>

            {/* Overview Tab */}
            {activeTab === 'overview' && (
              <div className="space-y-6">
                {/* Team Summary */}
                <div className="bg-white rounded-2xl shadow-lg p-6 border border-gray-200">
                  <h3 className="text-xl font-bold text-gray-900 mb-4">Team summary</h3>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="p-4 bg-blue-50 rounded-lg">
                      <div className="text-sm text-gray-600 mb-1">Team Value</div>
                      <div className="text-2xl font-bold text-blue-600">
                        £{analysisData?.team_analysis?.total_value || '0.0'}M
                      </div>
                    </div>
                    <div className="p-4 bg-green-50 rounded-lg">
                      <div className="text-sm text-gray-600 mb-1">Predicted Points</div>
                      <div className="text-2xl font-bold text-green-600">
                        {analysisData?.team_analysis?.predicted_points || '0.0'}
                      </div>
                    </div>
                    <div className="p-4 bg-purple-50 rounded-lg">
                      <div className="text-sm text-gray-600 mb-1">Team Strength</div>
                      <div className="text-2xl font-bold text-purple-600">
                        {analysisData?.team_analysis?.team_strength || '0.0'}/10
                      </div>
                    </div>
                    <div className="p-4 bg-yellow-50 rounded-lg">
                      <div className="text-sm text-gray-600 mb-1">Free Transfers</div>
                      <div className="text-2xl font-bold text-yellow-600">
                        {analysisData?.team_analysis?.free_transfers || 1}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Quick Actions */}
                <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-2xl p-6 border border-blue-200">
                  <h3 className="text-xl font-bold text-gray-900 mb-4">Quick actions</h3>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <button
                      onClick={() => setActiveTab('transfers')}
                      className="p-4 bg-white rounded-lg hover:shadow-lg transition-shadow text-left"
                    >
                      <div className="font-semibold text-gray-900">View Transfers</div>
                      <div className="text-sm text-gray-600">
                        {transferSuggestions.length || 0} suggestions
                      </div>
                    </button>
                    <button
                      onClick={() => setActiveTab('chips')}
                      className="p-4 bg-white rounded-lg hover:shadow-lg transition-shadow text-left"
                    >
                      <div className="font-semibold text-gray-900">Captain & Chips</div>
                      <div className="text-sm text-gray-600">
                        {captainCards.length || 0} options
                      </div>
                    </button>
                    <button
                      onClick={() => setActiveTab('bench')}
                      className="p-4 bg-white rounded-lg hover:shadow-lg transition-shadow text-left"
                    >
                      <div className="font-semibold text-gray-900">Optimize Bench</div>
                      <div className="text-sm text-gray-600">
                        Strength: {analysisData?.bench_analysis?.bench_strength || '0.0'}/10
                      </div>
                    </button>
                  </div>
                </div>

                {analysisData && (
                  <div className="grid gap-4 md:grid-cols-2">
                    <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                      <h4 className="text-sm font-semibold text-slate-900">Fixture outlook</h4>
                      <p className="mt-1 text-sm text-slate-500">Upcoming double and blank gameweeks.</p>
                      <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <p className="text-xs uppercase text-slate-500">Double weeks</p>
                          <p className="text-2xl font-semibold text-slate-900">{fixtureDoubleCount}</p>
                          {doubleGameweeks.slice(0, 2).map((gw: any) => (
                            <p key={gw?.gameweek ?? Math.random()} className="text-xs text-slate-500">
                              GW {gw?.gameweek ?? '-'} - {gw?.teams?.length ?? 0} teams
                            </p>
                          ))}
                        </div>
                        <div>
                          <p className="text-xs uppercase text-slate-500">Blank weeks</p>
                          <p className="text-2xl font-semibold text-slate-900">{fixtureBlankCount}</p>
                          {blankGameweeks.slice(0, 2).map((gw: any) => (
                            <p key={gw?.gameweek ?? Math.random()} className="text-xs text-slate-500">
                              GW {gw?.gameweek ?? '-'} - {gw?.teams_missing?.length ?? 0} teams idle
                            </p>
                          ))}
                        </div>
                      </div>
                    </div>
                    <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                      <h4 className="text-sm font-semibold text-slate-900">Market & news signals</h4>
                      <p className="mt-1 text-sm text-slate-500">Price trends and news-driven risks.</p>
                      <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <p className="text-xs uppercase text-slate-500">Price movement</p>
                          <p className="text-2xl font-semibold text-green-600">{risingCount}</p>
                          <p className="text-xs text-slate-500">{fallingCount} falling</p>
                        </div>
                        <div>
                          <p className="text-xs uppercase text-slate-500">High-risk players</p>
                          <p className="text-2xl font-semibold text-amber-600">{newsRiskCount}</p>
                          <p className="text-xs text-slate-500">{differentialCount} differential picks</p>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Transfers Tab */}
            {activeTab === 'transfers' && (
              <div className="bg-white rounded-2xl shadow-lg p-6 border border-gray-200">
                <h3 className="text-xl font-bold text-gray-900 mb-4">
                  Transfer recommendations
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6 text-sm">
                  <div className="p-4 rounded-xl border border-slate-200 bg-slate-50">
                    <div className="text-xs uppercase text-slate-500">Risk profile</div>
                    <div className="text-2xl font-semibold text-slate-900 mt-1">
                      {(preferences.risk_tolerance * 100).toFixed(0)}%
                    </div>
                    <p className="text-xs text-slate-500 mt-1">Prefers {preferences.formation_preference}</p>
                  </div>
                  <div className="p-4 rounded-xl border border-slate-200 bg-slate-50">
                    <div className="text-xs uppercase text-slate-500">Analysis modules</div>
                    <div className="flex flex-wrap gap-2 mt-2">
                      {[
                        { label: 'Fixture', enabled: analysisOptions.include_fixture_analysis },
                        { label: 'Price', enabled: analysisOptions.include_price_analysis },
                        { label: 'Strategy', enabled: analysisOptions.include_strategic_planning }
                      ].map((item) => (
                        <span
                          key={item.label}
                          className={`text-xs px-2 py-1 rounded-full border ${item.enabled ? 'bg-green-100 text-green-700 border-green-200' : 'bg-gray-100 text-gray-500 border-gray-200'
                            }`}
                        >
                          {item.label}
                        </span>
                      ))}
                    </div>
                  </div>
                  <div className="p-4 rounded-xl border border-slate-200 bg-slate-50">
                    <div className="text-xs uppercase text-slate-500">Resources</div>
                    <p className="mt-2">Free transfers: <span className="font-semibold text-slate-900">{freeTransfersDisplay}</span></p>
                    <p>Bank: <span className="font-semibold text-slate-900">£{bankDisplay}m</span></p>
                    <p>Chips considered: <span className="font-semibold text-slate-900">{chipsAvailableSet.size}</span></p>
                  </div>
                </div>

                <div className="space-y-4">
                  {transferSuggestions.length > 0 ? (
                    transferCardElements
                  ) : (
                    <div className="text-center py-8 text-gray-500">
                      <div className="text-4xl mb-4 text-slate-300">--</div>
                      <h4 className="text-lg font-medium text-gray-700 mb-2">No Transfer Suggestions</h4>
                      <p className="text-sm text-gray-500">Your team looks good for now!</p>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Captain & Chips Tab */}
            {activeTab === 'chips' && (
              <div className="space-y-6">
                {/* Captain Suggestions */}
                <div className="bg-white rounded-2xl shadow-lg p-6 border border-gray-200">
                  <h3 className="text-xl font-bold text-gray-900 mb-4">
                    Captain recommendations
                  </h3>

                  <div className="space-y-3">
                    {captainCards.length > 0 ? (
                      captainCards.map(({ suggestion, meta, adjustedScore }, index) => {
                        const difficulty = meta?.difficulty;
                        const opponentLabel = meta
                          ? `${meta.venue === 'home' ? 'vs' : '@'} ${meta.opponentName}`
                          : null;
                        const difficultyDescriptor =
                          typeof difficulty === 'number'
                            ? difficulty <= 2
                              ? 'Easy'
                              : difficulty >= 4
                                ? 'Hard'
                                : 'Medium'
                            : null;
                        return (
                          <div key={index} className="p-4 bg-gradient-to-r from-yellow-50 to-orange-50 rounded-xl border border-yellow-200">
                            <div className="flex items-center justify-between">
                              <div className="flex items-center space-x-3">
                                <div className="w-12 h-12 bg-yellow-100 rounded-full flex items-center justify-center">
                                  <span className="text-lg">{getPositionIcon(suggestion.player.position)}</span>
                                </div>
                                <div>
                                  <h4 className="font-bold text-gray-900">{suggestion.player.web_name}</h4>
                                  <p className="text-sm text-gray-600">{suggestion.player.team_name}</p>
                                  {opponentLabel && (
                                    <p className="text-xs text-gray-500">
                                      Next: {opponentLabel}
                                      {difficulty ? ` · Difficulty ${difficulty}/5${difficultyDescriptor ? ` (${difficultyDescriptor})` : ''}` : ''}
                                    </p>
                                  )}
                                </div>
                              </div>
                              <div className="text-right">
                                <div className="text-lg font-bold text-green-600">
                                  {adjustedScore.toFixed(1)} pts
                                </div>
                                <div className={`text-xs px-2 py-1 rounded-full ${suggestion.confidence === 'High' ? 'bg-green-100 text-green-800' :
                                  suggestion.confidence === 'Medium' ? 'bg-yellow-100 text-yellow-800' :
                                    'bg-red-100 text-red-800'
                                  }`}>
                                  {suggestion.confidence} confidence
                                </div>
                              </div>
                            </div>
                            <div className="mt-3 text-sm text-gray-700 bg-white/50 p-3 rounded-lg">
                              <strong>Why:</strong> {suggestion.reason}
                            </div>
                          </div>
                        );
                      })
                    ) : (
                      <div className="text-center py-8 text-gray-500">
                        <div className="text-4xl mb-4 text-slate-300">--</div>
                        <h4 className="text-lg font-medium text-gray-700 mb-2">No Captain Suggestions</h4>
                        <p className="text-sm text-gray-500">Analysis in progress...</p>
                      </div>
                    )}
                  </div>
                </div>

                {/* Chip Recommendations */}
                {chipOpportunities.length > 0 && (
                  <div className="bg-white rounded-2xl shadow-lg p-6 border border-gray-200">
                    <h3 className="text-xl font-bold text-gray-900 mb-4">Chip strategy</h3>
                    <div className="space-y-4">
                      {chipOpportunities.map((chip: any, index: number) => (
                        <div key={index} className={`p-4 rounded-xl border-2 ${chip.score >= 7 ? 'bg-gradient-to-r from-green-50 to-emerald-50 border-green-300' :
                          chip.score >= 5 ? 'bg-gradient-to-r from-yellow-50 to-orange-50 border-yellow-300' :
                            'bg-gradient-to-r from-gray-50 to-slate-50 border-gray-300'
                          }`}>
                          <div className="flex items-center justify-between mb-3">
                            <div className="flex items-center space-x-3">
                              <div className={`w-12 h-12 rounded-full flex items-center justify-center text-xs font-semibold uppercase ${chip.score >= 7 ? 'bg-green-200 text-green-900' :
                                chip.score >= 5 ? 'bg-yellow-200 text-yellow-900' :
                                  'bg-gray-200 text-gray-700'
                                }`}>
                                {chip.chip.replace(/\s+/g, '').slice(0, 2)}
                              </div>
                              <div>
                                <h4 className="font-bold text-gray-900">{chip.chip}</h4>
                                <p className="text-sm text-gray-600">
                                  Recommended: {chip.recommended_gameweek || 'Upcoming GW'}
                                </p>
                              </div>
                            </div>
                            <div className="text-right">
                              <div className="text-2xl font-bold text-purple-600">
                                {chip.score.toFixed(1)}/10
                              </div>
                              <div className={`text-xs px-2 py-1 rounded-full ${chip.confidence === 'High' ? 'bg-green-100 text-green-800' :
                                chip.confidence === 'Medium' ? 'bg-yellow-100 text-yellow-800' :
                                  'bg-red-100 text-red-800'
                                }`}>
                                {chip.confidence}
                              </div>
                            </div>
                          </div>
                          <p className="text-sm text-gray-700 mb-3">{chip.reason}</p>
                          {chip.conditions && chip.conditions.length > 0 && (
                            <div className="bg-white/80 p-3 rounded-lg">
                              <div className="text-xs font-semibold text-gray-700 mb-2">Conditions:</div>
                              <ul className="text-xs text-gray-600 space-y-1">
                                {chip.conditions.map((condition: string, i: number) => (
                                  <li key={i}>- {condition}</li>
                                ))}
                              </ul>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Bench & Formation Tab */}
            {activeTab === 'bench' && analysisData && (
              <div className="bg-white rounded-2xl shadow-lg p-6 border border-gray-200">
                <h3 className="text-xl font-bold text-gray-900 mb-4">
                  Bench & formation optimization
                </h3>
                {benchData && Object.keys(benchData).length > 0 ? (
                  <>
                    <div className="grid grid-cols-2 gap-4 mb-6">
                      <div className="p-4 bg-purple-50 rounded-lg border border-purple-200">
                        <div className="text-sm text-gray-600 mb-1">Bench Strength</div>
                        <div className="text-2xl font-bold text-purple-600">
                          {benchData?.bench_strength || 0}/10
                        </div>
                      </div>
                      <div className="p-4 bg-green-50 rounded-lg border border-green-200">
                        <div className="text-sm text-gray-600 mb-1">Bench Value</div>
                        <div className="text-2xl font-bold text-green-600">
                          £{benchData?.bench_value?.toFixed(1) || '0.0'}m
                        </div>
                      </div>
                    </div>

                    <div className="mb-6">
                      <h4 className="font-semibold text-gray-900 mb-3">Autosub recommendations</h4>
                      {autosubRecommendations.length > 0 ? (
                        <div className="space-y-3">
                          {autosubRecommendations.map((sub: any, index: number) => (
                            <div key={`${sub?.bench_player?.id ?? index}-${sub?.starter_to_replace?.id ?? index}`} className="p-3 bg-orange-50 rounded-lg border border-orange-200">
                              <div className="flex items-center justify-between mb-2">
                                <div className="flex items-center space-x-2 text-sm">
                                  <span className="text-green-700 font-semibold">{sub?.bench_player?.web_name}</span>
                                  <span className="text-gray-400">{'>'}</span>
                                  <span className="text-red-600 font-semibold">{sub?.starter_to_replace?.web_name}</span>
                                </div>
                                <div className="text-green-600 font-bold">+{sub?.points_gain} pts</div>
                              </div>
                              <p className="text-xs text-gray-600">{sub?.recommendation}</p>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <p className="rounded-lg border border-slate-200 bg-white p-3 text-xs text-slate-500">
                          No bench swaps recommended for this gameweek.
                        </p>
                      )}
                    </div>

                    {benchData?.bench_players?.length > 0 && (
                      <div className="mb-6">
                        <h4 className="font-semibold text-gray-900 mb-3">Bench players</h4>
                        <div className="grid grid-cols-2 gap-2">
                          {benchData.bench_players.map((player: any, index: number) => (
                            <div key={index} className="p-2 bg-gray-50 rounded-lg text-xs">
                              <div className="font-semibold text-gray-900">{player.web_name}</div>
                              <div className="text-gray-600">{player.team_name} - {player.position}</div>
                              <div className="text-gray-500">{player.predicted_points} pts - £{player.cost}m</div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {benchData?.recommendations?.length > 0 && (
                      <div>
                        <h4 className="font-semibold text-gray-900 mb-3">Recommendations</h4>
                        <div className="space-y-2">
                          {benchData.recommendations.map((rec: any, index: number) => (
                            <div key={index} className={`p-3 rounded-lg border ${rec.type === 'warning' ? 'bg-yellow-50 border-yellow-200' :
                              rec.type === 'chip' ? 'bg-purple-50 border-purple-200' :
                                rec.type === 'action' ? 'bg-red-50 border-red-200' :
                                  'bg-blue-50 border-blue-200'
                              }`}>
                              <div className="flex items-start justify-between">
                                <p className="text-sm text-gray-700 flex-1">{rec.message}</p>
                                <span className={`text-xs px-2 py-1 rounded-full ml-2 whitespace-nowrap ${rec.priority === 'High' ? 'bg-red-100 text-red-800' :
                                  rec.priority === 'Medium' ? 'bg-yellow-100 text-yellow-800' :
                                    'bg-blue-100 text-blue-800'
                                  }`}>
                                  {rec.priority}
                                </span>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </>
                ) : (
                  <div className="text-center text-gray-600 py-10">
                    <p className="text-lg font-semibold mb-2">No bench data yet</p>
                    <p className="text-sm">Select and submit your starting XI to unlock bench insights.</p>
                  </div>
                )}
              </div>
            )}

            {/* Advanced Tab */}
            {activeTab === 'advanced' && analysisData && (
              <div className="bg-white rounded-2xl shadow-lg p-6 border border-gray-200">
                <h3 className="text-xl font-bold text-gray-900 mb-4">Advanced analysis</h3>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">

                  {/* Fixture Analysis */}
                  {analysisData.fixture_analysis && (
                    <div className="p-4 bg-blue-50 rounded-lg border border-blue-200">
                      <h4 className="font-semibold text-blue-900 mb-2">Fixture analysis</h4>
                      <p className="text-sm text-blue-700">
                        {analysisData.fixture_analysis.double_gameweeks?.length || 0} double gameweeks detected
                      </p>
                      <p className="text-sm text-blue-700">
                        {analysisData.fixture_analysis.blank_gameweeks?.length || 0} blank gameweeks detected
                      </p>
                    </div>
                  )}

                  {/* Price Analysis */}
                  {analysisData.price_analysis && (
                    <div className="p-4 bg-green-50 rounded-lg border border-green-200">
                      <h4 className="font-semibold text-green-900 mb-2">Price analysis</h4>
                      <p className="text-sm text-green-700">
                        {analysisData.price_analysis.rising_players?.length || 0} players rising
                      </p>
                      <p className="text-sm text-green-700">
                        {analysisData.price_analysis.falling_players?.length || 0} players falling
                      </p>
                    </div>
                  )}

                  {/* Strategic Planning */}
                  {analysisData.strategic_planning && (
                    <div className="p-4 bg-purple-50 rounded-lg border border-purple-200">
                      <h4 className="font-semibold text-purple-900 mb-2">Strategic planning</h4>
                      <p className="text-sm text-purple-700">
                        {analysisData.strategic_planning.chip_opportunities?.length || 0} chip opportunities
                      </p>
                      <p className="text-sm text-purple-700">
                        Long-term strategy available
                      </p>
                    </div>
                  )}

                  {/* News Analysis */}
                  {analysisData.news_analysis && (
                    <div className="p-4 bg-orange-50 rounded-lg border border-orange-200">
                      <h4 className="font-semibold text-orange-900 mb-2">News analysis</h4>
                      <p className="text-sm text-orange-700">
                        {analysisData.news_analysis.high_risk_players?.length || 0} high-risk players
                      </p>
                      <p className="text-sm text-orange-700">
                        {analysisData.news_analysis.total_analyzed || 0} players analyzed
                      </p>
                    </div>
                  )}

                  {/* Effective Ownership */}
                  {analysisData.effective_ownership && (
                    <div className="p-4 bg-indigo-50 rounded-lg border border-indigo-200">
                      <h4 className="font-semibold text-indigo-900 mb-2">Effective ownership</h4>
                      <p className="text-sm text-indigo-700">
                        {analysisData.effective_ownership.template_team?.length || 0} template players
                      </p>
                      <p className="text-sm text-indigo-700">
                        {analysisData.effective_ownership.differential_opportunities?.length || 0} differentials
                      </p>
                    </div>
                  )}

                  {/* Model Status */}
                  <div className="p-4 bg-gray-50 rounded-lg border border-gray-200">
                    <h4 className="font-semibold text-gray-900 mb-2">Model usage</h4>
                    <p className="text-sm text-gray-700">
                      Ensemble models: {analysisData.features_used?.ensemble_models ? 'Enabled' : 'Disabled'}
                    </p>
                    <p className="text-sm text-gray-700">
                      Features: {analysisData.features_used?.total_features || 0}/6 active
                    </p>
                  </div>
                </div>

                {/* Warnings in Advanced Tab */}
                {analysisData?.warnings && analysisData.warnings.length > 0 && (
                  <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-4 mt-6">
                    <h4 className="text-lg font-bold text-yellow-900 mb-3">Analysis warnings</h4>
                    <div className="space-y-2">
                      {analysisData.warnings.map((warning, index) => (
                        <div key={index} className="text-sm text-yellow-800 bg-yellow-100 p-2 rounded">
                          {warning}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Action Buttons - Visible on all tabs */}
            <div className="bg-white rounded-2xl shadow-lg p-6 border border-gray-200">
              <h3 className="text-xl font-bold text-gray-900 mb-4">Next steps</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <button
                  onClick={() => window.open('https://fantasy.premierleague.com', '_blank')}
                  className="p-4 bg-gradient-to-r from-green-500 to-green-600 text-white rounded-xl hover:from-green-600 hover:to-green-700 transition-all duration-300 font-medium"
                >
                  Apply changes in FPL
                </button>
                <button
                  onClick={() => fetchAdvancedAnalysis(team)}
                  className="p-4 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-xl hover:from-blue-600 hover:to-blue-700 transition-all duration-300 font-medium"
                >
                  Refresh analysis
                </button>
                <button
                  onClick={() => router.push('/')}
                  className="p-4 bg-gradient-to-r from-gray-500 to-gray-600 text-white rounded-xl hover:from-gray-600 hover:to-gray-700 transition-all duration-300 font-medium"
                >
                  New team
                </button>
              </div>
            </div>
          </div>

          {/* Form Analysis Tab */}
          {activeTab === 'form' && (
            <div className="space-y-6">
              <div className="bg-white rounded-2xl shadow-lg p-6 border border-gray-200">
                <h3 className="text-xl font-bold text-gray-900 mb-2">Form Momentum Analysis</h3>
                <p className="text-gray-600 mb-6">
                  Statistical analysis of player form trends, hot/cold streaks, and breakout potential
                </p>
                <div className="space-y-4">
                  {team.slice(0, 11).map(player => (
                    <FormAnalyzer key={player.id} playerId={player.id} />
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Differentials Tab */}
          {activeTab === 'differentials' && (
            <div className="space-y-6">
              <DifferentialFinder />
            </div>
          )}

          {/* Injury Risk Tab */}
          {activeTab === 'injury' && (
            <div className="space-y-6">
              <InjuryRiskDashboard />
            </div>
          )}
        </div>
      </div>
    </div >
  );
}

function AnalysisLoading() {
  const steps = [
    { title: 'Validating squad', description: 'Checking formation and budget rules' },
    { title: 'Running models', description: 'Blending projections and ownership data' },
    { title: 'Preparing dashboard', description: 'Building transfer and captain insights' }
  ];
  const [activeStep, setActiveStep] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setActiveStep((prev) => (prev + 1) % steps.length);
    }, 2200);
    return () => clearInterval(timer);
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 via-white to-slate-100 flex items-center justify-center px-4">
      <div className="w-full max-w-3xl space-y-8 rounded-3xl bg-white/80 p-8 shadow-2xl ring-1 ring-slate-100">
        <div className="flex flex-col items-center gap-4 text-center">
          <div className="relative h-20 w-20">
            <div className="absolute inset-0 rounded-full bg-gradient-to-br from-blue-200 to-indigo-200 opacity-60 blur-2xl animate-pulse" />
            <div className="relative flex h-full w-full items-center justify-center rounded-full border-4 border-blue-100">
              <div className="loading-spinner h-10 w-10 rounded-full border-2 border-blue-400 border-t-indigo-500" />
            </div>
          </div>
          <div>
            <p className="text-sm uppercase tracking-[0.2em] text-blue-500">Running analysis</p>
            <h2 className="mt-2 text-3xl font-semibold text-slate-900">Preparing your optimization report</h2>
            <p className="mt-2 text-sm text-slate-500">Typical run time: 8–12 seconds. You can stay on this page while we crunch the numbers.</p>
          </div>
        </div>

        <div className="space-y-3">
          {steps.map((step, index) => (
            <div key={step.title} className="flex items-center gap-3">
              <div
                className={`h-8 w-8 flex items-center justify-center rounded-full border-2 ${index === activeStep
                  ? 'border-blue-500 bg-blue-50 text-blue-600'
                  : 'border-slate-200 bg-white text-slate-400'
                  }`}
              >
                {index + 1}
              </div>
              <div>
                <p className="text-sm font-semibold text-slate-900">{step.title}</p>
                <p className="text-xs text-slate-500">{step.description}</p>
              </div>
            </div>
          ))}
        </div>

        <div className="rounded-2xl bg-slate-50 p-4 text-sm text-slate-600">
          Tip: keep the tab open and we’ll automatically redirect once the results are ready.
        </div>
      </div>
    </div>
  );
}
