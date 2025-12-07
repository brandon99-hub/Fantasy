'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import dynamic from 'next/dynamic';
import { Player, DataStatus } from '@/lib/types';
import { FPLApi } from '@/lib/api';
import { saveAnalysisSession } from '@/lib/analysisSession';
import { useNotifications } from '@/components/ui/NotificationProvider';

// Lazy load TeamUpload component for faster initial page load
const TeamUpload = dynamic(() => import('@/components/TeamUpload'), {
  loading: () => (
    <div className="flex items-center justify-center py-16">
      <div className="loading-spinner h-10 w-10 rounded-full border-2 border-primary-200 border-t-primary-500"></div>
    </div>
  ),
  ssr: false
});

export default function FPLTool() {
  const [loading, setLoading] = useState(false);
  const [pageReady, setPageReady] = useState(false);
  const router = useRouter();
  const [statusError, setStatusError] = useState<string | null>(null);
  const { notify } = useNotifications();
  
  // Mark page as ready immediately after mount
  useEffect(() => {
    setPageReady(true);
  }, []);

  const handleTeamUpload = async (data: { players: Player[], config: any }) => {
    setLoading(true);
    
    try {
      const sessionId = saveAnalysisSession({
        players: data.players,
        config: data.config
      });
      router.push(`/analysis?session=${sessionId}`);
    } catch (error) {
      console.error('Error redirecting to analysis:', error);
      notify('We were unable to process your team. Please try again.', {
        title: 'Navigation error',
        variant: 'error'
      });
    } finally {
      setLoading(false);
    }
  };

  const [dataStatus, setDataStatus] = useState<DataStatus | null>(null);

  // Lazy load data status only when needed
  useEffect(() => {
    let isMounted = true;

    const initializeApp = async () => {
      try {
        const status = await FPLApi.getDataStatus();
        if (isMounted) {
          setDataStatus(status);
          setStatusError(null);
        }
      } catch (error) {
        console.error('Failed to fetch data status', error);
        if (isMounted) {
          setDataStatus({
            has_data: false,
            player_count: 0,
            message: 'Unable to verify data'
          });
          setStatusError('Unable to reach backend');
        }
      }
    };
    
    // Delay initial load slightly to prioritize UI render
    const timer = setTimeout(initializeApp, 100);
    return () => {
      isMounted = false;
      clearTimeout(timer);
    };
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 via-white to-slate-100">
      <header className={`sticky top-0 z-40 border-b border-slate-200 bg-white/90 backdrop-blur-md transition-opacity duration-300 ${pageReady ? 'opacity-100' : 'opacity-0'}`}>
        <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-4 sm:px-6">
          <div className="flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-to-br from-blue-600 to-indigo-500 text-base font-semibold text-white shadow-lg">
              FPL
            </div>
            <div>
              <p className="text-xs uppercase tracking-wide text-slate-500">Fantasy Premier League</p>
              <h1 className="text-xl font-semibold text-slate-900">AI Performance Studio</h1>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <StatusBadge dataStatus={dataStatus} error={statusError} />
            <button
              onClick={() => router.push('/analysis')}
              className="hidden rounded-lg border border-slate-200 px-3 py-2 text-sm font-medium text-slate-700 transition-colors hover:bg-slate-50 sm:block"
            >
              View analysis
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-4 py-10 sm:px-6">
        <section className="surface-card mb-8 p-6 sm:p-8">
          <div className="grid gap-8 lg:grid-cols-3">
            <div className="space-y-6 lg:col-span-2">
              <div>
                <p className="text-sm font-semibold uppercase tracking-wide text-blue-600">Plan smarter in minutes</p>
                <h2 className="mt-2 text-3xl font-semibold text-slate-900 sm:text-4xl">
                  Upload your FPL squad and unlock data-backed transfer plans.
            </h2>
                <p className="mt-3 text-base text-slate-600 sm:text-lg">
                  Blend player projections, ownership trends, fixture difficulty, and chip strategy so you can focus on making winning calls each gameweek.
                </p>
              </div>
              <DataPoints dataStatus={dataStatus} />
            </div>
            <div className="space-y-4 rounded-2xl border border-slate-100 bg-slate-50/70 p-5">
              <h3 className="text-sm font-semibold text-slate-800">Why managers trust this studio</h3>
              <ul className="space-y-3 text-sm text-slate-600">
                <li className="flex items-start gap-2">
                  <span className="mt-1 h-2 w-2 rounded-full bg-blue-500" />
                  Live squad validation with club and position caps.
                </li>
                <li className="flex items-start gap-2">
                  <span className="mt-1 h-2 w-2 rounded-full bg-indigo-500" />
                  Trusted projections for transfers, chips, and captaincy picks.
                </li>
                <li className="flex items-start gap-2">
                  <span className="mt-1 h-2 w-2 rounded-full bg-slate-500" />
                  Bench insights plus fixture planning for the next six weeks.
                </li>
              </ul>
            </div>
          </div>
        </section>
          
        <section className="surface-card p-4 sm:p-6">
          <TeamUpload onTeamUpload={handleTeamUpload} loading={loading} />
        </section>
      </main>
        </div>
  );
}

function StatusBadge({ dataStatus, error }: { dataStatus: DataStatus | null; error: string | null }) {
  if (!dataStatus) {
    return (
      <div className="flex items-center gap-2 rounded-full border border-slate-200 px-3 py-1 text-sm text-slate-500">
        <span className="loading-spinner h-3 w-3 rounded-full border border-blue-400 border-t-transparent"></span>
        Checking data...
      </div>
    );
  }

  const variant = error ? 'error' : dataStatus.has_data ? 'success' : 'warning';
  const label =
    variant === 'success'
      ? `${dataStatus.player_count.toLocaleString()} players synced`
      : variant === 'warning'
      ? 'Data missing'
      : 'Backend offline';

  const colors: Record<string, string> = {
    success: 'bg-green-50 text-green-700 border-green-200',
    warning: 'bg-yellow-50 text-yellow-700 border-yellow-200',
    error: 'bg-red-50 text-red-700 border-red-200'
  };

  return (
    <div className={`flex items-center gap-2 rounded-full border px-3 py-1 text-sm font-medium ${colors[variant]}`}>
      <span
        className={`h-2 w-2 rounded-full ${
          variant === 'success' ? 'bg-green-500' : variant === 'warning' ? 'bg-yellow-500' : 'bg-red-500'
        }`}
      />
      {label}
    </div>
  );
}

function DataPoints({ dataStatus }: { dataStatus: DataStatus | null }) {
  return (
    <div className="grid gap-4 sm:grid-cols-3">
      <InsightTile
        title="Player database"
        value={dataStatus?.player_count ? dataStatus.player_count.toLocaleString() : '—'}
        caption={dataStatus?.message ?? 'Awaiting status'}
      />
      <InsightTile
        title="Optimizer"
        value="Advanced AI"
        caption="Combines projections + EO + fixtures"
      />
      <InsightTile
        title="Chip planner"
        value="Bench Boost & Free Hit"
        caption="Recommendations ready"
      />
    </div>
  );
}

function InsightTile({ title, value, caption }: { title: string; value: string; caption: string }) {
  return (
    <div className="rounded-2xl border border-slate-100 bg-white/80 p-4 shadow-sm">
      <p className="text-xs uppercase tracking-wide text-slate-500">{title}</p>
      <p className="mt-2 text-2xl font-semibold text-slate-900">{value}</p>
      <p className="text-sm text-slate-500">{caption}</p>
    </div>
  );
}