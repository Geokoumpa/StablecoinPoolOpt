'use client';

import { useDashboard } from '@/hooks/api-hooks';
import { transformDashboardData } from '@/lib/data-transformations';
import { formatCurrency, formatPercentage, formatNumber } from '@/lib/fetch-utils';

export default function DashboardPage() {
    const { data: dashboardData, isLoading, error, refetch } = useDashboard();

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
                            onClick={() => refetch()}
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
                        onClick={() => refetch()}
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
                    <div className="bg-white shadow rounded-lg p-6">
                        <h3 className="text-lg font-medium text-gray-900 mb-2">Total Pools</h3>
                        <p className="text-3xl font-bold text-blue-600">
                            {transformedData.totalPoolsFormatted}
                        </p>
                        <p className="text-sm text-gray-500">Active pools</p>
                    </div>

                    <div className="bg-white shadow rounded-lg p-6">
                        <h3 className="text-lg font-medium text-gray-900 mb-2">Optimization Runs</h3>
                        <p className="text-3xl font-bold text-green-600">
                            {transformedData.totalOptimizationRunsFormatted}
                        </p>
                        <p className="text-sm text-gray-500">Total runs</p>
                    </div>

                    <div className="bg-white shadow rounded-lg p-6">
                        <h3 className="text-lg font-medium text-gray-900 mb-2">Total Value Locked</h3>
                        <p className="text-3xl font-bold text-purple-600">
                            {transformedData.totalValueLockedFormatted}
                        </p>
                        <p className="text-sm text-gray-500">Across all pools</p>
                    </div>

                    <div className="bg-white shadow rounded-lg p-6">
                        <h3 className="text-lg font-medium text-gray-900 mb-2">Average APY</h3>
                        <p className="text-3xl font-bold text-orange-600">
                            {transformedData.averageAPYFormatted}
                        </p>
                        <p className="text-sm text-gray-500">Weighted average</p>
                    </div>
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