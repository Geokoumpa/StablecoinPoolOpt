import { useState, useEffect, useCallback, useRef } from 'react';
import { GetServerSideProps, GetStaticProps, GetStaticPaths } from 'next';
import { prisma } from './db';

// Types for Next.js data fetching
export interface NextDataFetchingOptions {
    revalidate?: number;
    notFound?: boolean;
    redirect?: {
        destination: string;
        permanent: boolean;
    };
}

// Generic server-side props function
export function createGetServerSideProps<T>(
    dataFetcher: () => Promise<T>,
    options: NextDataFetchingOptions = {}
): GetServerSideProps<{ data: T | null; error?: string }> {
    return async (context) => {
        try {
            const data = await dataFetcher();

            if (options.notFound && !data) {
                return {
                    notFound: true,
                };
            }

            if (options.redirect) {
                return {
                    redirect: options.redirect,
                };
            }

            return {
                props: {
                    data,
                },
            };
        } catch (error) {
            console.error('Server-side data fetching error:', error);

            return {
                props: {
                    data: null,
                    error: error instanceof Error ? error.message : 'Unknown error',
                },
            };
        }
    };
}

// Generic static props function
export function createGetStaticProps<T>(
    dataFetcher: () => Promise<T>,
    options: NextDataFetchingOptions = {}
): GetStaticProps<{ data: T | null; error?: string }> {
    return async () => {
        try {
            const data = await dataFetcher();

            if (options.notFound && !data) {
                return {
                    notFound: true,
                };
            }

            if (options.redirect) {
                return {
                    redirect: options.redirect,
                };
            }

            return {
                props: {
                    data,
                },
                revalidate: options.revalidate,
            };
        } catch (error) {
            console.error('Static data fetching error:', error);

            return {
                props: {
                    data: null,
                    error: error instanceof Error ? error.message : 'Unknown error',
                },
                revalidate: options.revalidate || 60, // Default to 1 minute
            };
        }
    };
}

// Generic static paths function for dynamic routes
export function createGetStaticPaths(
    pathsFetcher: () => Promise<{ params: { [key: string]: string }[] }>
): GetStaticPaths {
    return async () => {
        try {
            const pathsResult = await pathsFetcher();

            return {
                paths: pathsResult.params.map(pathParams => ({
                    params: pathParams,
                })),
                fallback: false,
            };
        } catch (error) {
            console.error('Static paths generation error:', error);

            return {
                paths: [],
                fallback: false,
            };
        }
    };
}

// Specific data fetchers for different entities
export const dashboardDataFetcher = async () => {
    // Fetch dashboard metrics
    const totalPools = await prisma.pools.count();
    const totalRuns = await prisma.asset_allocations.count();
    const totalValueLocked = await prisma.pools.aggregate({
        _sum: { tvl: true },
    });
    const averageAPY = await prisma.pools.aggregate({
        _avg: { apy: true },
    });

    return {
        totalPools,
        totalOptimizationRuns: totalRuns,
        totalValueLocked: totalValueLocked._sum.tvl || 0,
        averageAPY: averageAPY._avg.apy || 0,
    };
};

export const poolsDataFetcher = async (params: {
    page?: number;
    limit?: number;
    search?: string;
    protocol?: string;
    chain?: string;
}) => {
    const { page = 1, limit = 10, search, protocol, chain } = params;
    const skip = (page - 1) * limit;

    const where: any = {};

    if (search) {
        where.OR = [
            { name: { contains: search, mode: 'insensitive' } },
            { protocol: { contains: search, mode: 'insensitive' } },
        ];
    }

    if (protocol) {
        where.protocol = protocol;
    }

    if (chain) {
        where.chain = chain;
    }

    const [pools, totalCount] = await Promise.all([
        prisma.pools.findMany({
            where,
            skip,
            take: limit,
            orderBy: { last_updated: 'desc' },
        }),
        prisma.pools.count({ where }),
    ]);

    return {
        data: pools,
        pagination: {
            page,
            limit,
            totalCount,
            totalPages: Math.ceil(totalCount / limit),
            hasNextPage: page < Math.ceil(totalCount / limit),
            hasPreviousPage: page > 1,
        },
    };
};

export const poolDataFetcher = async (poolId: string) => {
    const pool = await prisma.pools.findUnique({
        where: { pool_id: poolId },
        include: {
            pool_daily_metrics: {
                take: 30,
                orderBy: { date: 'desc' },
            },
        },
    });

    return pool;
};

export const optimizationRunsDataFetcher = async (params: {
    page?: number;
    limit?: number;
}) => {
    const { page = 1, limit = 10 } = params;
    const skip = (page - 1) * limit;

    const [runs, totalCount] = await Promise.all([
        prisma.asset_allocations.findMany({
            skip,
            take: limit,
            orderBy: { timestamp: 'desc' },
        }),
        prisma.asset_allocations.count(),
    ]);

    return {
        data: runs,
        pagination: {
            page,
            limit,
            totalCount,
            totalPages: Math.ceil(totalCount / limit),
            hasNextPage: page < Math.ceil(totalCount / limit),
            hasPreviousPage: page > 1,
        },
    };
};

export const optimizationRunDataFetcher = async (runId: string) => {
    // Find first record with this run_id since we need to id for the unique constraint
    const run = await prisma.asset_allocations.findFirst({
        where: { run_id: runId },
    });

    return run;
};

export const tokensDataFetcher = async (params: {
    page?: number;
    limit?: number;
    type?: 'approved' | 'blacklisted';
}) => {
    const { page = 1, limit = 10, type = 'approved' } = params;
    const skip = (page - 1) * limit;

    const table = type === 'approved' ? 'approved_tokens' : 'blacklisted_tokens';

    // Note: This is a simplified example. In practice, you'd need to handle
    // different token tables appropriately
    const tokens = await prisma.$queryRaw`
    SELECT * FROM ${table} 
    ORDER BY added_timestamp DESC 
    LIMIT ${limit} OFFSET ${skip}
  `;

    const totalCount = await prisma.$queryRaw`
    SELECT COUNT(*) as count FROM ${table}
  `;

    return {
        data: tokens,
        pagination: {
            page,
            limit,
            totalCount: (totalCount as any)[0].count,
            totalPages: Math.ceil((totalCount as any)[0].count / limit),
            hasNextPage: page < Math.ceil((totalCount as any)[0].count / limit),
            hasPreviousPage: page > 1,
        },
    };
};

export const protocolsDataFetcher = async (params: {
    page?: number;
    limit?: number;
}) => {
    const { page = 1, limit = 10 } = params;
    const skip = (page - 1) * limit;

    const [protocols, totalCount] = await Promise.all([
        prisma.approved_protocols.findMany({
            skip,
            take: limit,
            orderBy: { added_timestamp: 'desc' },
        }),
        prisma.approved_protocols.count(),
    ]);

    return {
        data: protocols,
        pagination: {
            page,
            limit,
            totalCount,
            totalPages: Math.ceil(totalCount / limit),
            hasNextPage: page < Math.ceil(totalCount / limit),
            hasPreviousPage: page > 1,
        },
    };
};

export const configurationParametersFetcher = async () => {
    const parameters = await prisma.default_allocation_parameters.findMany({
        orderBy: { parameter_name: 'asc' },
    });

    // Group parameters by category
    const groupedParameters = parameters.reduce((acc: any, param: any) => {
        const category = getParameterCategory(param.parameter_name);
        if (!acc[category]) {
            acc[category] = [];
        }
        acc[category].push(param);
        return acc;
    }, {} as Record<string, any[]>);

    return {
        parameters: groupedParameters,
        rawParameters: parameters,
    };
};

// Helper function to categorize parameters
function getParameterCategory(parameterName: string): string {
    if (parameterName.includes('gas')) return 'Gas Settings';
    if (parameterName.includes('allocation')) return 'Allocation Settings';
    if (parameterName.includes('filter')) return 'Filter Settings';
    if (parameterName.includes('threshold')) return 'Threshold Settings';
    if (parameterName.includes('limit')) return 'Limit Settings';
    if (parameterName.includes('wallet')) return 'Wallet Settings';
    if (parameterName.includes('protocol')) return 'Protocol Settings';
    if (parameterName.includes('token')) return 'Token Settings';

    return 'General Settings';
}

// Pre-built data fetching functions for common use cases
export const getDashboardServerSideProps = createGetServerSideProps(dashboardDataFetcher);

export const getPoolsStaticProps = (params: any) =>
    createGetStaticProps(() => poolsDataFetcher(params), {
        revalidate: 60, // Revalidate every minute
    });

export const getPoolStaticProps = (poolId: string) =>
    createGetStaticProps(() => poolDataFetcher(poolId), {
        revalidate: 300, // Revalidate every 5 minutes
        notFound: true,
    });

export const getOptimizationRunsStaticProps = (params: any) =>
    createGetStaticProps(() => optimizationRunsDataFetcher(params), {
        revalidate: 60,
    });

export const getOptimizationRunStaticProps = (runId: string) =>
    createGetStaticProps(() => optimizationRunDataFetcher(runId), {
        revalidate: 300,
        notFound: true,
    });

export const getTokensStaticProps = (params: any) =>
    createGetStaticProps(() => tokensDataFetcher(params), {
        revalidate: 300,
    });

export const getProtocolsStaticProps = (params: any) =>
    createGetStaticProps(() => protocolsDataFetcher(params), {
        revalidate: 600, // Revalidate every 10 minutes
    });

export const getConfigurationStaticProps = createGetStaticProps(
    configurationParametersFetcher,
    {
        revalidate: 600,
    }
);

// Static paths generators for dynamic routes
export const getPoolsStaticPaths = createGetStaticPaths(async () => {
    const pools = await prisma.pools.findMany({
        select: { pool_id: true },
        take: 100, // Limit to first 100 for static generation
    });

    return {
        params: pools.map((pool: any) => ({
            poolId: pool.pool_id,
        })),
    };
});

export const getOptimizationRunsStaticPaths = createGetStaticPaths(async () => {
    const runs = await prisma.asset_allocations.findMany({
        select: { run_id: true },
        take: 100, // Limit to first 100 for static generation
    });

    return {
        params: runs.map((run: any) => ({
            runId: run.run_id,
        })),
    };
});

// Client-side data fetching with SWR-like pattern
export function useClientData<T>(
    key: string,
    fetcher: () => Promise<T>,
    options: {
        revalidateOnFocus?: boolean;
        revalidateOnReconnect?: boolean;
        refreshInterval?: number;
        dedupingInterval?: number;
    } = {}
) {
    const [data, setData] = useState<T | null>(null);
    const [error, setError] = useState<Error | null>(null);
    const [isLoading, setIsLoading] = useState(true);

    const {
        revalidateOnFocus = true,
        revalidateOnReconnect = true,
        refreshInterval = 0,
        dedupingInterval = 2000,
    } = options;

    const fetchRef = useRef<boolean>(false);
    const lastFetchRef = useRef<number>(0);

    const fetchData = useCallback(async () => {
        const now = Date.now();

        // Deduping logic
        if (fetchRef.current && now - lastFetchRef.current < dedupingInterval) {
            return;
        }

        setIsLoading(true);
        setError(null);
        fetchRef.current = true;
        lastFetchRef.current = now;

        try {
            const result = await fetcher();
            setData(result);
            return result;
        } catch (err) {
            setError(err as Error);
            throw err;
        } finally {
            setIsLoading(false);
            fetchRef.current = false;
        }
    }, [fetcher, dedupingInterval]);

    // Initial fetch
    useEffect(() => {
        fetchData();
    }, [fetchData]);

    // Refresh interval
    useEffect(() => {
        if (refreshInterval > 0) {
            const interval = setInterval(fetchData, refreshInterval);
            return () => clearInterval(interval);
        }
    }, [fetchData, refreshInterval]);

    // Revalidate on focus
    useEffect(() => {
        if (revalidateOnFocus) {
            const handleFocus = () => fetchData();
            window.addEventListener('focus', handleFocus);
            return () => window.removeEventListener('focus', handleFocus);
        }
    }, [fetchData, revalidateOnFocus]);

    // Revalidate on reconnect
    useEffect(() => {
        if (revalidateOnReconnect) {
            const handleOnline = () => fetchData();
            window.addEventListener('online', handleOnline);
            return () => window.removeEventListener('online', handleOnline);
        }
    }, [fetchData, revalidateOnReconnect]);

    return {
        data,
        error,
        isLoading,
        mutate: setData,
        revalidate: fetchData,
    };
}