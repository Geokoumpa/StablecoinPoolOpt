import { useState, useEffect, useCallback, useRef, useMemo } from 'react';

// Types for API responses
export interface ApiResponse<T> {
    data?: T;
    error?: string;
    isLoading: boolean;
    pagination?: {
        page: number;
        limit: number;
        totalCount: number;
        totalPages: number;
        hasNextPage: boolean;
        hasPreviousPage: boolean;
    };
}

export interface PaginationParams {
    page?: number;
    limit?: number;
    search?: string;
    protocol?: string;
    chain?: string;
    days?: number;
    sortBy?: string;
    sortOrder?: 'asc' | 'desc';
}

// Base fetch function with error handling and caching
async function apiFetch<T>(
    url: string,
    options: RequestInit = {},
    cacheKey?: string
): Promise<{ data?: T; error?: string }> {
    try {
        // Check cache first if cacheKey is provided
        if (cacheKey && typeof window !== 'undefined') {
            const cached = localStorage.getItem(cacheKey);
            if (cached) {
                const { data, timestamp } = JSON.parse(cached);
                // Cache for 5 minutes by default
                if (Date.now() - timestamp < 5 * 60 * 1000) {
                    return { data };
                }
            }
        }

        const response = await fetch(url, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers,
            },
            ...options,
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        // Cache response if cacheKey is provided
        if (cacheKey && typeof window !== 'undefined') {
            localStorage.setItem(cacheKey, JSON.stringify({
                data,
                timestamp: Date.now(),
            }));
        }

        return { data };
    } catch (error) {
        return {
            error: error instanceof Error ? error.message : 'An unknown error occurred'
        };
    }
}

// Generic hook for data fetching
export function useApi<T>(
    url: string | null,
    options: RequestInit = {},
    cacheKey?: string,
    dependencies: any[] = []
): ApiResponse<T> & { refetch: () => void } {
    const [state, setState] = useState<ApiResponse<T>>({
        data: undefined,
        error: undefined,
        isLoading: false,
    });

    // Memoize options to prevent infinite re-renders
    const memoizedOptions = useMemo(() => options, [JSON.stringify(options)]);

    const fetchData = useCallback(async () => {
        if (!url) return;

        setState(prev => ({ ...prev, isLoading: true, error: undefined }));

        const result = await apiFetch<T>(url, memoizedOptions, cacheKey);

        setState({
            data: result.data,
            error: result.error,
            isLoading: false,
            pagination: (result.data as any)?.pagination,
        });
    }, [url, memoizedOptions, cacheKey]);

    useEffect(() => {
        fetchData();
    }, [fetchData, ...dependencies]);

    // Refetch function
    const refetch = useCallback(() => {
        if (cacheKey && typeof window !== 'undefined') {
            localStorage.removeItem(cacheKey);
        }
        fetchData();
    }, [cacheKey, fetchData]);

    return {
        ...state,
        refetch,
    };
}

// Hook for paginated data
export function usePaginatedApi<T>(
    baseUrl: string,
    initialParams: PaginationParams = {},
    cacheKeyPrefix?: string
) {
    const [params, setParams] = useState<PaginationParams>(initialParams);

    const queryString = new URLSearchParams(
        Object.entries(params).filter(([_, value]) => value !== undefined) as [string, string][]
    ).toString();

    const url = `${baseUrl}${queryString ? `?${queryString}` : ''}`;
    const cacheKey = cacheKeyPrefix ? `${cacheKeyPrefix}_${queryString}` : undefined;

    const result = useApi<T>(url, {}, cacheKey, [queryString]);

    const updateParams = useCallback((newParams: Partial<PaginationParams>) => {
        setParams(prev => ({ ...prev, ...newParams }));
    }, []);

    const resetParams = useCallback(() => {
        setParams(initialParams);
    }, [initialParams]);

    return {
        ...result,
        params,
        updateParams,
        resetParams,
    };
}

// Hook for real-time data with polling
export function useRealTimeApi<T>(
    url: string | null,
    options: RequestInit = {},
    intervalMs: number = 30000, // 30 seconds default
    enabled: boolean = true
): ApiResponse<T> & { stopPolling: () => void; startPolling: () => void } {
    const [state, setState] = useState<ApiResponse<T>>({
        data: undefined,
        error: undefined,
        isLoading: false,
    });

    const intervalRef = useRef<NodeJS.Timeout | null>(null);

    // Memoize options to prevent infinite re-renders
    const memoizedOptions = useMemo(() => options, [JSON.stringify(options)]);

    const fetchData = useCallback(async () => {
        if (!url) return;

        setState(prev => ({ ...prev, isLoading: true, error: undefined }));

        const result = await apiFetch<T>(url, memoizedOptions);

        setState({
            data: result.data,
            error: result.error,
            isLoading: false,
            pagination: (result.data as any)?.pagination,
        });
    }, [url, memoizedOptions]);

    const startPolling = useCallback(() => {
        if (intervalRef.current) {
            clearInterval(intervalRef.current);
        }

        if (enabled && url) {
            fetchData(); // Initial fetch
            intervalRef.current = setInterval(fetchData, intervalMs);
        }
    }, [enabled, url, intervalMs, fetchData]);

    const stopPolling = useCallback(() => {
        if (intervalRef.current) {
            clearInterval(intervalRef.current);
            intervalRef.current = null;
        }
    }, []);

    useEffect(() => {
        startPolling();
        return stopPolling;
    }, [startPolling, stopPolling]);

    return {
        ...state,
        stopPolling,
        startPolling,
    };
}

// Mutation hook for POST/PUT/DELETE operations
export function useMutation<TData, TVariables = any>(
    url: string,
    method: 'POST' | 'PUT' | 'DELETE' = 'POST',
    options: RequestInit = {}
) {
    const [state, setState] = useState<{
        data?: TData;
        error?: string;
        isLoading: boolean;
    }>({
        data: undefined,
        error: undefined,
        isLoading: false,
    });

    // Memoize options to prevent infinite re-renders
    const memoizedOptions = useMemo(() => options, [JSON.stringify(options)]);

    const mutate = useCallback(async (variables?: TVariables) => {
        setState(prev => ({ ...prev, isLoading: true, error: undefined }));

        const result = await apiFetch<TData>(url, {
            method,
            body: variables ? JSON.stringify(variables) : undefined,
            ...memoizedOptions,
        });

        setState({
            data: result.data,
            error: result.error,
            isLoading: false,
        });

        return result;
    }, [url, method, memoizedOptions]);

    const reset = useCallback(() => {
        setState({
            data: undefined,
            error: undefined,
            isLoading: false,
        });
    }, []);

    return {
        ...state,
        mutate,
        reset,
    };
}

// Utility to clear cache
export function clearCache(keyPattern?: string) {
    if (typeof window === 'undefined') return;

    if (keyPattern) {
        Object.keys(localStorage).forEach(key => {
            if (key.includes(keyPattern)) {
                localStorage.removeItem(key);
            }
        });
    } else {
        // Clear all API cache
        Object.keys(localStorage).forEach(key => {
            if (key.startsWith('api_')) {
                localStorage.removeItem(key);
            }
        });
    }
}

// Utility to format currency
export function formatCurrency(amount: number | string, currency = 'USD'): string {
    const num = typeof amount === 'string' ? parseFloat(amount) : amount;
    if (isNaN(num)) return '0';
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency,
        minimumFractionDigits: 0,
        maximumFractionDigits: 2,
    }).format(num);
}

// Utility to format percentage
export function formatPercentage(value: number | string): string {
    const num = typeof value === 'string' ? parseFloat(value) : value;
    if (isNaN(num)) return '0.00%';
    return `${num.toFixed(2)}%`;
}

// Utility to format large numbers
export function formatNumber(num: number | string): string {
    const n = typeof num === 'string' ? parseFloat(num) : num;
    if (isNaN(n)) return '0';

    if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
    if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
    if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;

    return n.toFixed(0);
}

// Utility to format date
export function formatDate(date: string | Date): string {
    const d = typeof date === 'string' ? new Date(date) : date;
    if (isNaN(d.getTime())) return 'Invalid Date';
    return d.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
    });
}

// Utility to format date time
export function formatDateTime(date: string | Date): string {
    const d = typeof date === 'string' ? new Date(date) : date;
    if (isNaN(d.getTime())) return 'Invalid Date';
    return d.toLocaleString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
    });
}