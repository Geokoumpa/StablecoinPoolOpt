import { NextRequest, NextResponse } from 'next/server';
import { prisma } from '@/lib/db';
import { handleApiError, createSuccessResponse, getQueryParams, getDateRange } from '@/lib/api-utils';

export async function GET(request: NextRequest) {
    try {
        const { days, protocol, chain } = getQueryParams(request);
        const { startDate, endDate } = getDateRange(days);

        // Build where clause for daily metrics
        const where: any = {
            date: {
                gte: startDate,
                lte: endDate
            },
            pool: {
                currently_filtered_out: false
            }
        };

        if (protocol) {
            where.pool.protocol = protocol;
        }

        if (chain) {
            where.pool.chain = chain;
        }

        // Get aggregated metrics by date
        const metrics = await prisma.poolDailyMetrics.groupBy({
            by: ['date'],
            where,
            _avg: {
                actual_apy: true,
                forecasted_apy: true,
                actual_tvl: true,
                forecasted_tvl: true
            },
            _sum: {
                actual_tvl: true,
                forecasted_tvl: true
            },
            _count: {
                pool_id: true
            },
            orderBy: {
                date: 'asc'
            }
        });

        // Get top performing pools
        const topPools = await prisma.poolDailyMetrics.findMany({
            where,
            take: 10,
            orderBy: {
                actual_apy: 'desc'
            },
            include: {
                pool: {
                    select: {
                        name: true,
                        protocol: true,
                        chain: true,
                        symbol: true
                    }
                }
            }
        });

        // Get protocol performance summary
        const protocolSummary = await prisma.poolDailyMetrics.groupBy({
            by: ['date'],
            where,
            _avg: {
                actual_apy: true
            }
        });

        const response = createSuccessResponse({
            metrics,
            topPools,
            protocolSummary,
            period: {
                startDate,
                endDate,
                days
            }
        }, 300, 60);

        return response;
    } catch (error) {
        return handleApiError(error, 'Failed to fetch pool metrics');
    }
}