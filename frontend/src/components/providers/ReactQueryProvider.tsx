'use client';

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactNode, useState } from 'react';

export function ReactQueryProvider({ children }: { children: ReactNode }) {
    const [queryClient] = useState(
        () =>
            new QueryClient({
                defaultOptions: {
                    queries: {
                        staleTime: 0,  // Always consider data stale
                        gcTime: 1000 * 60 * 5,  // Cache for 5 minutes (renamed from cacheTime)
                        refetchOnWindowFocus: true,  // Refetch when window regains focus
                        refetchOnMount: true,  // Always refetch on component mount
                        retry: 1,  // Retry failed requests once
                    },
                },
            })
    );

    return (
        <QueryClientProvider client={queryClient}>
            {children}
        </QueryClientProvider>
    );
}
