import { formatCurrency, formatPercentage, formatNumber, formatDate, formatDateTime } from './fetch-utils';

// Transform pool data for display
export function transformPoolData(pool: any) {
    return {
        ...pool,
        tvlFormatted: formatCurrency(pool.tvl),
        apyFormatted: formatPercentage(pool.apy),
        createdAtFormatted: formatDate(pool.created_at),
        updatedAtFormatted: formatDate(pool.updated_at),
        status: pool.currently_filtered_out ? 'Filtered Out' : 'Active',
        statusColor: pool.currently_filtered_out ? 'text-red-600' : 'text-green-600',
    };
}

// Transform pool metrics for charts
export function transformPoolMetrics(metrics: any[]) {
    return metrics.map(metric => ({
        ...metric,
        date: formatDate(metric.date),
        tvlFormatted: formatCurrency(metric.tvl),
        apyFormatted: formatPercentage(metric.apy),
        volume24hFormatted: formatCurrency(metric.volume_24h),
        tvlNumber: parseFloat(metric.tvl) || 0,
        apyNumber: parseFloat(metric.apy) || 0,
        volume24hNumber: parseFloat(metric.volume_24h) || 0,
    }));
}

// Transform optimization run data
export function transformOptimizationRun(run: any) {
    return {
        ...run,
        createdAtFormatted: formatDateTime(run.created_at),
        updatedAtFormatted: formatDateTime(run.updated_at),
        statusFormatted: run.status.replace('_', ' ').replace(/\b\w/g, (l: string) => l.toUpperCase()),
        statusColor: getStatusColor(run.status),
        parametersSummary: getParametersSummary(run.parameters),
        resultsSummary: getResultsSummary(run.results),
    };
}

// Transform token data
export function transformTokenData(token: any) {
    return {
        ...token,
        createdAtFormatted: formatDate(token.created_at),
        addressFormatted: formatAddress(token.address),
        chainFormatted: token.chain.toUpperCase(),
    };
}

// Transform protocol data
export function transformProtocolData(protocol: any) {
    return {
        ...protocol,
        createdAtFormatted: formatDate(protocol.created_at),
        chainFormatted: protocol.chain.toUpperCase(),
        descriptionFormatted: protocol.description || 'No description available',
    };
}

// Transform configuration parameters
export function transformConfigurationParameters(parameters: any[]) {
    return parameters.map(param => ({
        ...param,
        createdAtFormatted: formatDate(param.created_at),
        updatedAtFormatted: formatDate(param.updated_at),
        valueFormatted: formatParameterValue(param.parameter_value),
        category: getParameterCategory(param.parameter_name),
    }));
}

// Transform dashboard data
export function transformDashboardData(data: any) {
    return {
        ...data,
        totalValueLockedFormatted: formatCurrency(data.totalValueLocked),
        averageAPYFormatted: formatPercentage(data.averageAPY),
        totalPoolsFormatted: formatNumber(data.totalPools),
        totalOptimizationRunsFormatted: formatNumber(data.totalOptimizationRuns),
    };
}

// Helper functions
function getStatusColor(status: string): string {
    switch (status.toLowerCase()) {
        case 'completed':
            return 'text-green-600';
        case 'running':
            return 'text-blue-600';
        case 'failed':
            return 'text-red-600';
        case 'pending':
            return 'text-yellow-600';
        default:
            return 'text-gray-600';
    }
}

function getParametersSummary(parameters: any): string {
    if (!parameters) return 'No parameters';

    const keys = Object.keys(parameters);
    if (keys.length === 0) return 'No parameters';

    return `${keys.length} parameter${keys.length > 1 ? 's' : ''}`;
}

function getResultsSummary(results: any): string {
    if (!results) return 'No results';

    if (typeof results === 'object' && results.total_return) {
        return `Return: ${formatPercentage(results.total_return)}`;
    }

    return 'Results available';
}

function formatAddress(address: string): string {
    if (!address || address.length < 10) return address;

    return `${address.slice(0, 6)}...${address.slice(-4)}`;
}

function formatParameterValue(value: any): string {
    if (typeof value === 'number') {
        return formatNumber(value);
    }

    if (typeof value === 'boolean') {
        return value ? 'Yes' : 'No';
    }

    if (typeof value === 'object' && value !== null) {
        return JSON.stringify(value, null, 2);
    }

    return String(value);
}

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

// Chart data transformation functions
export function transformMetricsForChart(metrics: any[], metricType: 'tvl' | 'apy' | 'volume') {
    return metrics.map(item => ({
        date: formatDate(item.date),
        value: parseFloat(item[metricType]) || 0,
        formattedValue: metricType === 'apy'
            ? formatPercentage(item[metricType])
            : formatCurrency(item[metricType]),
    }));
}

export function transformPoolDistributionData(pools: any[]) {
    const protocolDistribution: Record<string, number> = {};

    pools.forEach(pool => {
        const protocol = pool.protocol || 'Unknown';
        protocolDistribution[protocol] = (protocolDistribution[protocol] || 0) + parseFloat(pool.tvl) || 0;
    });

    return Object.entries(protocolDistribution).map(([protocol, tvl]) => ({
        name: protocol,
        value: tvl,
        formattedValue: formatCurrency(tvl),
        percentage: (tvl / Object.values(protocolDistribution).reduce((a, b) => a + b, 0)) * 100,
    }));
}

export function transformAPYDistributionData(pools: any[]) {
    const ranges = [
        { name: '0-2%', min: 0, max: 2 },
        { name: '2-5%', min: 2, max: 5 },
        { name: '5-10%', min: 5, max: 10 },
        { name: '10-20%', min: 10, max: 20 },
        { name: '20%+', min: 20, max: Infinity },
    ];

    return ranges.map(range => ({
        name: range.name,
        value: pools.filter(pool => {
            const apy = parseFloat(pool.apy) || 0;
            return apy >= range.min && apy < range.max;
        }).length,
    }));
}

// Export data transformation functions
export function exportToCSV(data: any[], filename: string) {
    if (!data || data.length === 0) return;

    const headers = Object.keys(data[0]);
    const csvContent = [
        headers.join(','),
        ...data.map(row =>
            headers.map(header => {
                const value = row[header];
                if (typeof value === 'string' && value.includes(',')) {
                    return `"${value}"`;
                }
                return value;
            }).join(',')
        )
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${filename}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}

export function exportToJSON(data: any[], filename: string) {
    if (!data || data.length === 0) return;

    const jsonContent = JSON.stringify(data, null, 2);
    const blob = new Blob([jsonContent], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${filename}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}