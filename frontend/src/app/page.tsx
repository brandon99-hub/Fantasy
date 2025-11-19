'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import dynamic from 'next/dynamic';
import { Player } from '@/lib/types';

// Lazy load TeamUpload component for faster initial page load
const TeamUpload = dynamic(() => import('@/components/TeamUpload'), {
  loading: () => (
    <div className="flex items-center justify-center py-20">
      <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
    </div>
  ),
  ssr: false
});

export default function FPLTool() {
  const [loading, setLoading] = useState(false);
  const [pageReady, setPageReady] = useState(false);
  const router = useRouter();
  
  // Mark page as ready immediately after mount
  useEffect(() => {
    setPageReady(true);
  }, []);

  const handleTeamUpload = async (data: { players: Player[], config: any }) => {
    setLoading(true);
    
    try {
      const teamData = encodeURIComponent(JSON.stringify(data.players));
      const configData = encodeURIComponent(JSON.stringify(data.config));
      router.push(`/analysis?team=${teamData}&config=${configData}`);
    } catch (error) {
      console.error('Error redirecting to analysis:', error);
      alert('Error processing team. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const [dataStatus, setDataStatus] = useState<any>(null);

  // Lazy load data status only when needed
  useEffect(() => {
    const initializeApp = async () => {
      try {
        // Only check data status, don't pre-load anything
        const statusResponse = await fetch('http://localhost:8000/api/system/data-status', {
          signal: AbortSignal.timeout(5000) // 5 second timeout
        });
        if (statusResponse.ok) {
          const status = await statusResponse.json();
          setDataStatus(status);
        }
      } catch (error) {
        // Fail silently, show fallback UI
        setDataStatus({ has_data: true, player_count: 0 });
      }
    };
    
    // Delay initial load slightly to prioritize UI render
    const timer = setTimeout(initializeApp, 100);
    return () => clearTimeout(timer);
  }, []);

  return (
    <div className="min-h-screen">
      {/* Enhanced Header */}
      <div className={`glass sticky top-0 z-50 border-b border-white/20 transition-opacity duration-300 ${pageReady ? 'opacity-100' : 'opacity-0'}`}>
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="w-12 h-12 gradient-primary rounded-xl flex items-center justify-center shadow-medium">
                <span className="text-2xl">⚽</span>
              </div>
              <div>
                <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                  FPL AI Assistant
                </h1>
                <p className="text-sm text-gray-600">Smart Fantasy Premier League optimization</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-6">
              {/* Enhanced Data Status */}
              {dataStatus ? (
                <div className="flex items-center space-x-3 bg-white/80 rounded-xl px-4 py-2 shadow-soft">
                  <div className={`w-3 h-3 rounded-full status-indicator ${
                    dataStatus.has_data ? 'status-online' : 'status-offline'
                  }`}></div>
                  <div className="text-sm">
                    <div className="font-medium text-gray-900">
                      {dataStatus.player_count > 0 ? `${dataStatus.player_count} players` : 'Ready'}
                    </div>
                    <div className="text-xs text-gray-500">
                      Database ready
                    </div>
                  </div>
                </div>
              ) : (
                <div className="flex items-center space-x-2 bg-white/80 rounded-xl px-4 py-2 shadow-soft">
                  <div className="loading-spinner w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full"></div>
                  <span className="text-sm text-gray-600">Loading...</span>
                </div>
              )}
              
            </div>
          </div>
        </div>
      </div>

      {/* Main Content with enhanced spacing */}
      <div className="max-w-7xl mx-auto px-6 py-12">
        <div className="space-y-8">
          {/* Hero Section */}
          <div className="text-center mb-12">
            <h2 className="text-4xl font-bold text-white mb-4">
              Upload Your FPL Team
            </h2>
            <p className="text-xl text-white/80 max-w-2xl mx-auto">
              Get AI-powered transfer suggestions and captain recommendations to maximize your points
            </p>
          </div>
          
          <TeamUpload onTeamUpload={handleTeamUpload} loading={loading} />
        </div>
      </div>
    </div>
  );
}