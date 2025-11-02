import { NextRequest, NextResponse } from 'next/server';
import { prisma } from '@/lib/db';
import { auth } from '@clerk/nextjs/server';

export async function GET(request: NextRequest) {
    try {
        // Check authentication
        const { userId } = await auth();

        if (!userId) {
            return NextResponse.json(
                { error: 'Unauthorized - Please sign in to access dashboard data' },
                { status: 401 }
            );
        }
        // Get total pools count
        const totalPools = await prisma.pools.count({
            where: {
                currently_filtered_out: false
            }
        });

        // Get optimization runs count
        const optimizationRuns = await prisma.allocation_parameters.count();

        // Get total TVL
        const tvlResult = await prisma.pools.aggregate({
            where: {
                currently_filtered_out: false,
                tvl: {
                    not: null
                }
            },
            _sum: {
                tvl: true
            }
        });

        // Get average APY
        const apyResult = await prisma.pools.aggregate({
            where: {
                currently_filtered_out: false,
                apy: {
                    not: null
                }
            },
            _avg: {
                apy: true
            }
        });

        // Get recent activity (last 5 optimization runs)
        const recentActivity = await prisma.allocation_parameters.findMany({
            take: 5,
            orderBy: {
                timestamp: 'desc'
            },
            select: {
                run_id: true,
                timestamp: true,
                projected_apy: true,
                transaction_costs: true
            }
        });

        const dashboardData = {
            totalPools,
            optimizationRuns,
            totalTVL: tvlResult._sum.tvl || 0,
            averageAPY: apyResult._avg.apy || 0,
            recentActivity
        };

        // Add cache headers
        const response = NextResponse.json(dashboardData);
        response.headers.set('Cache-Control', 'public, s-maxage=60, stale-while-revalidate=30');

        return response;
    } catch (error) {
        console.error('Dashboard API error:', error);
        return NextResponse.json(
            { error: 'Failed to fetch dashboard data' },
            { status: 500 }
        );
    }
}