'use client';

import { useState } from 'react';
import { usePools } from '@/hooks/api-hooks';
import { transformPoolData } from '@/lib/data-transformations';
import { formatCurrency, formatPercentage } from '@/lib/fetch-utils';

export default function PoolsIndex() {
    const { data: poolsData, isLoading, error, refetch, params, updateParams } = usePools({
        page: 1,
        limit: 10,
    });

    const [searchTerm, setSearchTerm] = useState('');
    const [protocolFilter, setProtocolFilter] = useState('');

    const handleSearch = () => {
        updateParams({
            search: searchTerm,
            protocol: protocolFilter,
            page: 1, // Reset to first page when searching
        });
    };

    const handleClearFilters = () => {
        setSearchTerm('');
        setProtocolFilter('');
        updateParams({
            search: '',
            protocol: '',
            page: 1,
        });
    };

    const handlePageChange = (newPage: number) => {
        updateParams({ page: newPage });
    };

    if (isLoading && !poolsData) {
        return (
            <div className="space-y-6">
                <div className="bg-white shadow rounded-lg p-6">
                    <div className="animate-pulse">
                        <div className="h-8 bg-gray-200 rounded w-1/4 mb-4"></div>
                        <div className="h-4 bg-gray-200 rounded w-1/2"></div>
                    </div>
                </div>

                <div className="bg-white shadow rounded-lg">
                    <div className="px-4 py-5 sm:p-6">
                        <div className="animate-pulse">
                            <div className="h-10 bg-gray-200 rounded w-full mb-4"></div>
                            <div className="overflow-hidden shadow ring-1 ring-black ring-opacity-5 md:rounded-lg">
                                <div className="min-w-full py-2 align-middle md:px-6 lg:px-8">
                                    {[...Array(5)].map((_, i) => (
                                        <div key={i} className="animate-pulse py-4 border-b border-gray-200">
                                            <div className="h-4 bg-gray-200 rounded w-1/4 mb-2"></div>
                                            <div className="h-4 bg-gray-200 rounded w-1/6"></div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="space-y-6">
                <div className="bg-white shadow rounded-lg p-6">
                    <div className="text-red-600">
                        <h2 className="text-xl font-semibold mb-2">Error Loading Pools</h2>
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

    const pools = poolsData?.data || [];
    const pagination = poolsData?.pagination;

    return (
        <div className="space-y-6">
            <div className="bg-white shadow rounded-lg p-6">
                <h1 className="text-2xl font-bold text-gray-900">Pools Management</h1>
                <p className="mt-2 text-gray-600">
                    Manage and monitor stablecoin pools across different protocols
                </p>
            </div>

            <div className="bg-white shadow rounded-lg">
                <div className="px-4 py-5 sm:p-6">
                    <div className="sm:flex sm:items-center">
                        <div className="sm:flex-auto">
                            <h3 className="text-lg leading-6 font-medium text-gray-900">Pools</h3>
                            <p className="mt-2 text-sm text-gray-700">
                                A list of all stablecoin pools including their TVL, APY, and protocol information.
                            </p>
                        </div>
                        <div className="mt-4 sm:mt-0 sm:ml-16 sm:flex-none">
                            <button
                                type="button"
                                className="inline-flex items-center justify-center rounded-md border border-transparent bg-blue-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 sm:w-auto"
                            >
                                <svg className="-ml-1 mr-2 h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                                </svg>
                                Add Pool
                            </button>
                        </div>
                    </div>

                    {/* Filters */}
                    <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">Search</label>
                            <input
                                type="text"
                                value={searchTerm}
                                onChange={(e) => setSearchTerm(e.target.value)}
                                placeholder="Search by name or protocol..."
                                className="block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
                            />
                        </div>
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">Protocol</label>
                            <input
                                type="text"
                                value={protocolFilter}
                                onChange={(e) => setProtocolFilter(e.target.value)}
                                placeholder="Filter by protocol..."
                                className="block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
                            />
                        </div>
                        <div className="flex items-end space-x-2">
                            <button
                                onClick={handleSearch}
                                className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
                            >
                                Search
                            </button>
                            <button
                                onClick={handleClearFilters}
                                className="px-4 py-2 bg-gray-300 text-gray-700 rounded-md hover:bg-gray-400 focus:outline-none focus:ring-2 focus:ring-gray-500"
                            >
                                Clear
                            </button>
                        </div>
                    </div>

                    <div className="mt-8 flex flex-col">
                        <div className="-my-2 -mx-4 overflow-x-auto sm:-mx-6 lg:-mx-8">
                            <div className="inline-block min-w-full py-2 align-middle md:px-6 lg:px-8">
                                <div className="overflow-hidden shadow ring-1 ring-black ring-opacity-5 md:rounded-lg">
                                    <table className="min-w-full divide-y divide-gray-300">
                                        <thead className="bg-gray-50">
                                            <tr>
                                                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                                    Pool Name
                                                </th>
                                                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                                    Protocol
                                                </th>
                                                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                                    TVL
                                                </th>
                                                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                                    APY
                                                </th>
                                                <th scope="col" className="relative px-6 py-3">
                                                    <span className="sr-only">Actions</span>
                                                </th>
                                            </tr>
                                        </thead>
                                        <tbody className="bg-white divide-y divide-gray-200">
                                            {pools.length === 0 ? (
                                                <tr>
                                                    <td colSpan={5} className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 text-center">
                                                        No pools found
                                                    </td>
                                                </tr>
                                            ) : (
                                                pools.map((pool: any) => {
                                                    const transformedPool = transformPoolData(pool);
                                                    return (
                                                        <tr key={pool.pool_id}>
                                                            <td className="px-6 py-4 whitespace-nowrap">
                                                                <div className="text-sm font-medium text-gray-900">{pool.name}</div>
                                                                <div className="text-sm text-gray-500">{pool.chain}</div>
                                                            </td>
                                                            <td className="px-6 py-4 whitespace-nowrap">
                                                                <div className="text-sm text-gray-900">{pool.protocol}</div>
                                                            </td>
                                                            <td className="px-6 py-4 whitespace-nowrap">
                                                                <div className="text-sm text-gray-900">{transformedPool.tvlFormatted}</div>
                                                            </td>
                                                            <td className="px-6 py-4 whitespace-nowrap">
                                                                <div className={`text-sm font-medium ${transformedPool.apy > 5 ? 'text-green-600' : 'text-gray-900'}`}>
                                                                    {transformedPool.apyFormatted}
                                                                </div>
                                                            </td>
                                                            <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                                                                <button className="text-blue-600 hover:text-blue-900">
                                                                    View
                                                                </button>
                                                            </td>
                                                        </tr>
                                                    );
                                                })
                                            )}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>

                        {/* Pagination */}
                        {pagination && pagination.totalPages > 1 && (
                            <div className="mt-6 flex items-center justify-between">
                                <div className="text-sm text-gray-700">
                                    Showing {((pagination.page - 1) * pagination.limit) + 1} to{' '}
                                    {Math.min(pagination.page * pagination.limit, pagination.totalCount)} of{' '}
                                    {pagination.totalCount} results
                                </div>
                                <div className="flex space-x-2">
                                    <button
                                        onClick={() => handlePageChange(pagination.page - 1)}
                                        disabled={!pagination.hasPreviousPage}
                                        className="px-3 py-1 text-sm border rounded-md disabled:opacity-50 disabled:cursor-not-allowed"
                                    >
                                        Previous
                                    </button>
                                    <span className="px-3 py-1 text-sm">
                                        Page {pagination.page} of {pagination.totalPages}
                                    </span>
                                    <button
                                        onClick={() => handlePageChange(pagination.page + 1)}
                                        disabled={!pagination.hasNextPage}
                                        className="px-3 py-1 text-sm border rounded-md disabled:opacity-50 disabled:cursor-not-allowed"
                                    >
                                        Next
                                    </button>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}