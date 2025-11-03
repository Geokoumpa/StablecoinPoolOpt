import { useApi, usePaginatedApi, useMutation, PaginationParams } from '@/lib/fetch-utils';

// Types based on API responses
export interface DashboardData {
    totalPools: number;
    totalOptimizationRuns: number;
    totalValueLocked: number;
    averageAPY: number;
    recentActivity: any[];
}

export interface ChartData {
    poolAPYDistribution: Array<{
        name: string;
        apy: number;
    }>;
    tvlByProtocol: Array<{
        name: string;
        tvl: number;
        percentage: number;
    }>;
    optimizationTrends: Array<{
        date: string;
        projected_apy: number;
        transaction_costs: number;
        run_id: string;
    }>;
}

export interface Pool {
    pool_id: string;
    pool_name: string;
    protocol: string;
    chain: string;
    tvl: number;
    apy: number;
    currently_filtered_out: boolean;
    created_at: string;
    updated_at: string;
    daily_metrics?: PoolDailyMetric[];
}

export interface PoolDailyMetric {
    id: number;
    pool_id: string;
    date: string;
    tvl: number;
    apy: number;
    volume_24h: number;
    created_at: string;
}

export interface OptimizationRun {
    id: string;
    status: string;
    parameters: any;
    results: any;
    created_at: string;
    updated_at: string;
}

export interface Token {
    id: string;
    symbol: string;
    name: string;
    address: string;
    decimals: number;
    chain: string;
    created_at: string;
}

export interface Protocol {
    id: string;
    name: string;
    chain: string;
    description?: string;
    created_at: string;
}

export interface AllocationParameter {
    id: number;
    parameter_name: string;
    parameter_value: any;
    description?: string;
    created_at: string;
    updated_at: string;
}

// Dashboard hooks
export function useDashboard() {
    return useApi<DashboardData>('/api/dashboard', {}, 'api_dashboard');
}

export function useDashboardCharts() {
    return useApi<ChartData>('/api/dashboard/charts', {}, 'api_dashboard_charts');
}

// Pools hooks
export function usePools(initialParams: PaginationParams = {}) {
    return usePaginatedApi<{ data: Pool[]; pagination: any }>(
        '/api/pools/list',
        initialParams,
        'api_pools'
    );
}

export function usePool(poolId: string | null) {
    return useApi<Pool>(poolId ? `/api/pools/${poolId}` : null, {}, `api_pool_${poolId}`);
}

export function usePoolMetrics(initialParams: PaginationParams = {}) {
    return usePaginatedApi<{ data: any[]; pagination: any }>(
        '/api/pools/metrics',
        initialParams,
        'api_pool_metrics'
    );
}

// Optimization runs hooks
export function useOptimizationRuns(initialParams: PaginationParams = {}) {
    return usePaginatedApi<{ data: OptimizationRun[]; pagination: any }>(
        '/api/optimization/runs',
        initialParams,
        'api_optimization_runs'
    );
}

export function useOptimizationRun(runId: string | null) {
    return useApi<OptimizationRun>(
        runId ? `/api/optimization/runs/${runId}` : null,
        {},
        `api_optimization_run_${runId}`
    );
}

// Tokens hooks
export function useApprovedTokens(initialParams: PaginationParams = {}) {
    return usePaginatedApi<{ data: Token[]; pagination: any }>(
        '/api/tokens/approved',
        initialParams,
        'api_approved_tokens'
    );
}

export function useBlacklistedTokens(initialParams: PaginationParams = {}) {
    return usePaginatedApi<{ data: Token[]; pagination: any }>(
        '/api/tokens/blacklisted',
        initialParams,
        'api_blacklisted_tokens'
    );
}

// Protocols hooks
export function useApprovedProtocols(initialParams: PaginationParams = {}) {
    return usePaginatedApi<{ data: Protocol[]; pagination: any }>(
        '/api/protocols/approved',
        initialParams,
        'api_approved_protocols'
    );
}

// Configuration hooks
export function useConfigurationParameters() {
    return useApi<{ parameters: Record<string, AllocationParameter[]>; rawParameters: AllocationParameter[] }>(
        '/api/config/parameters',
        {},
        'api_config_parameters'
    );
}

export function useWalletAddresses() {
    return useApi<any>('/api/config/wallets', {}, 'api_wallet_addresses');
}

// Mutation hooks
export function useCreatePool() {
    return useMutation<Pool, Partial<Pool>>('/api/pools', 'POST');
}

export function useUpdatePool(poolId: string) {
    return useMutation<Pool, Partial<Pool>>(`/api/pools/${poolId}`, 'PUT');
}

export function useDeletePool(poolId: string) {
    return useMutation<void, void>(`/api/pools/${poolId}`, 'DELETE');
}

export function useCreateToken() {
    return useMutation<Token, Partial<Token>>('/api/tokens/approved', 'POST');
}

export function useUpdateToken(tokenId: string) {
    return useMutation<Token, Partial<Token>>(`/api/tokens/approved/${tokenId}`, 'PUT');
}

export function useDeleteToken(tokenId: string) {
    return useMutation<void, void>(`/api/tokens/approved/${tokenId}`, 'DELETE');
}

export function useBlacklistToken() {
    return useMutation<Token, Partial<Token>>('/api/tokens/blacklisted', 'POST');
}

export function useCreateProtocol() {
    return useMutation<Protocol, Partial<Protocol>>('/api/protocols/approved', 'POST');
}

export function useUpdateProtocol(protocolId: string) {
    return useMutation<Protocol, Partial<Protocol>>(`/api/protocols/approved/${protocolId}`, 'PUT');
}

export function useDeleteProtocol(protocolId: string) {
    return useMutation<void, void>(`/api/protocols/approved/${protocolId}`, 'DELETE');
}

export function useUpdateConfigurationParameters() {
    return useMutation<any, any>('/api/config/parameters', 'PUT');
}

export function useUpdateWalletAddresses() {
    return useMutation<any, any>('/api/config/wallets', 'PUT');
}

// Utility function to build query strings
export function buildQueryString(params: Record<string, any>): string {
    const searchParams = new URLSearchParams();

    Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined && value !== null && value !== '') {
            searchParams.append(key, String(value));
        }
    });

    return searchParams.toString();
}

// Utility function to handle API errors
export function handleApiError(error: string | undefined, toast: any) {
    if (error) {
        toast({
            title: 'Error',
            description: error,
            variant: 'destructive',
        });
    }
}

// Utility function to handle API success
export function handleApiSuccess(message: string, toast: any) {
    toast({
        title: 'Success',
        description: message,
    });
}