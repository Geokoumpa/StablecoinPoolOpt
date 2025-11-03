'use client';

import { useDashboard, useDashboardCharts } from '@/hooks/api-hooks';
import { transformDashboardData } from '@/lib/data-transformations';
import { formatCurrency, formatPercentage, formatNumber } from '@/lib/fetch-utils';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { PoolAPYDistribution } from '@/components/charts/pool-apy-distribution';
import { TVLByProtocol } from '@/components/charts/tvl-by-protocol';
import { OptimizationTrends } from '@/components/charts/optimization-trends';

export default function DashboardPage() {
    const { data: dashboardData, isLoading: dashboardLoading, error: dashboardError, refetch: refetchDashboard } = useDashboard();
    const { data: chartData, isLoading: chartsLoading, error: chartsError, refetch: refetchCharts } = useDashboardCharts();

    const isLoading = dashboardLoading || chartsLoading;
    const error = dashboardError || chartsError;

    const handleRefresh = () => {
        refetchDashboard();
        refetchCharts();
    };

    if (isLoading) {
        return (
            <div className="space-y-6">
                <div className="bg-white shadow rounded-lg p-6">
                    <div className="animate-pulse">
                        <div className="h-8 bg-gray-200 rounded w-1/4 mb-4"></div>
                        <div className="h-4 bg-gray-200 rounded w-1/2"></div>
                    </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                    {[...Array(4)].map((_, i) => (
                        <div key={i} className="bg-white shadow rounded-lg p-6">
                            <div className="animate-pulse">
                                <div className="h-6 bg-gray-200 rounded w-1/2 mb-2"></div>
                                <div className="h-10 bg-gray-200 rounded w-3/4 mb-2"></div>
                                <div className="h-4 bg-gray-200 rounded w-1/3"></div>
                            </div>
                        </div>
                    ))}
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {[...Array(3)].map((_, i) => (
                        <div key={i} className="bg-white shadow rounded-lg p-6">
                            <div className="animate-pulse">
                                <div className="h-6 bg-gray-200 rounded w-1/3 mb-4"></div>
                                <div className="h-64 bg-gray-200 rounded"></div>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="space-y-6">
                <div className="bg-white shadow rounded-lg p-6">
                    <div className="text-red-600">
                        <h2 className="text-xl font-semibold mb-2">Error Loading Dashboard</h2>
                        <p>{error}</p>
                        <button
                            onClick={handleRefresh}
                            className="mt-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
                        >
                            Try Again
                        </button>
                    </div>
                </div>
            </div>
        );
    }

    const transformedData = dashboardData ? transformDashboardData(dashboardData) : null;

    return (
        <div className="space-y-6">
            <div className="bg-white shadow rounded-lg p-6">
                <div className="flex justify-between items-center">
                    <div>
                        <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
                        <p className="mt-2 text-gray-600">
                            Welcome to the Stablecoin Pool Optimization admin interface
                        </p>
                    </div>
                    <button
                        onClick={handleRefresh}
                        className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 flex items-center gap-2"
                    >
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                        </svg>
                        Refresh
                    </button>
                </div>
            </div>

            {transformedData && (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                    <Card>
                        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                            <CardTitle className="text-sm font-medium">Total Pools</CardTitle>
                            <svg className="h-4 w-4 text-muted-foreground" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
                            </svg>
                        </CardHeader>
                        <CardContent>
                            <div className="text-2xl font-bold text-blue-600">
                                {transformedData.totalPoolsFormatted}
                            </div>
                            <p className="text-xs text-muted-foreground">Active pools</p>
                        </CardContent>
                    </Card>

                    <Card>
                        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                            <CardTitle className="text-sm font-medium">Optimization Runs</CardTitle>
                            <svg className="h-4 w-4 text-muted-foreground" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                            </svg>
                        </CardHeader>
                        <CardContent>
                            <div className="text-2xl font-bold text-green-600">
                                {transformedData.totalOptimizationRunsFormatted}
                            </div>
                            <p className="text-xs text-muted-foreground">Total runs</p>
                        </CardContent>
                    </Card>

                    <Card>
                        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                            <CardTitle className="text-sm font-medium">Total Value Locked</CardTitle>
                            <svg className="h-4 w-4 text-muted-foreground" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                            </svg>
                        </CardHeader>
                        <CardContent>
                            <div className="text-2xl font-bold text-purple-600">
                                {transformedData.totalValueLockedFormatted}
                            </div>
                            <p className="text-xs text-muted-foreground">Across all pools</p>
                        </CardContent>
                    </Card>

                    <Card>
                        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                            <CardTitle className="text-sm font-medium">Average APY</CardTitle>
                            <svg className="h-4 w-4 text-muted-foreground" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                            </svg>
                        </CardHeader>
                        <CardContent>
                            <div className="text-2xl font-bold text-orange-600">
                                {transformedData.averageAPYFormatted}
                            </div>
                            <p className="text-xs text-muted-foreground">Weighted average</p>
                        </CardContent>
                    </Card>
                </div>
            )}

            {/* Charts Section */}
            {chartData && (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <Card>
                        <CardHeader>
                            <CardTitle>Pool APY Distribution</CardTitle>
                            <CardDescription>Top 10 pools by APY</CardDescription>
                        </CardHeader>
                        <CardContent>
                            <PoolAPYDistribution data={chartData.poolAPYDistribution} />
                        </CardContent>
                    </Card>

                    <Card>
                        <CardHeader>
                            <CardTitle>TVL by Protocol</CardTitle>
                            <CardDescription>Total value locked across protocols</CardDescription>
                        </CardHeader>
                        <CardContent>
                            <TVLByProtocol data={chartData.tvlByProtocol} />
                        </CardContent>
                    </Card>

                    <Card className="lg:col-span-2">
                        <CardHeader>
                            <CardTitle>Optimization Trends</CardTitle>
                            <CardDescription>Recent optimization runs performance</CardDescription>
                        </CardHeader>
                        <CardContent>
                            <OptimizationTrends data={chartData.optimizationTrends} />
                        </CardContent>
                    </Card>
                </div>
            )}

            <div className="bg-white shadow rounded-lg p-6">
                <h2 className="text-lg font-medium text-gray-900 mb-4">Recent Activity</h2>
                {transformedData && transformedData.totalPools > 0 ? (
                    <div className="text-center py-8 text-green-600">
                        <svg className="mx-auto h-12 w-12 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        <p className="mt-2">System is actively monitoring pools</p>
                        <p className="text-sm text-gray-500 mt-1">
                            Last updated: {new Date().toLocaleTimeString()}
                        </p>
                    </div>
                ) : (
                    <div className="text-center py-8 text-gray-500">
                        <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        <p className="mt-2">No recent activity</p>
                        <p className="text-sm text-gray-500 mt-1">
                            {transformedData ? 'Pools will appear here once data is available' : 'Loading data...'}
                        </p>
                    </div>
                )}
            </div>
        </div>
    );
}