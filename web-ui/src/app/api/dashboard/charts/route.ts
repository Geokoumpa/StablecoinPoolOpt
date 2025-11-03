import { NextRequest, NextResponse } from 'next/server';
import { prisma } from '@/lib/db';
import { auth } from '@clerk/nextjs/server';

export async function GET(request: NextRequest) {
    try {
        // Check authentication
        const { userId } = await auth();

        if (!userId) {
            return NextResponse.json(
                { error: 'Unauthorized - Please sign in to access chart data' },
                { status: 401 }
            );
        }

        // Get pool APY distribution data (top 10 pools by APY)
        const poolAPYData = await prisma.pools.findMany({
            where: {
                currently_filtered_out: false,
                apy: {
                    not: null,
                    gt: 0
                }
            },
            select: {
                pool_address: true,
                apy: true,
                protocol: true
            },
            orderBy: {
                apy: 'desc'
            },
            take: 10
        });

        const formattedAPYData = poolAPYData.map(pool => ({
            name: pool.protocol || (pool.pool_address ? pool.pool_address.slice(0, 8) + '...' : 'Unknown'),
            apy: pool.apy ? parseFloat(pool.apy.toString()) : 0
        }));

        // Get TVL by protocol data
        const tvlByProtocolData = await prisma.pools.groupBy({
            by: ['protocol'],
            where: {
                currently_filtered_out: false,
                tvl: {
                    not: null,
                    gt: 0
                }
            },
            _sum: {
                tvl: true
            },
            orderBy: {
                _sum: {
                    tvl: 'desc'
                }
            }
        });

        // Filter out null protocols and calculate total TVL
        const filteredTVLData = tvlByProtocolData.filter(item => item.protocol !== null);
        const totalTVL = filteredTVLData.reduce((sum, item) => {
            const tvl = item._sum?.tvl || 0;
            return sum + (typeof tvl === 'number' ? tvl : parseFloat(tvl.toString()));
        }, 0);

        const formattedTVLData = filteredTVLData.map(item => ({
            name: item.protocol || 'Unknown',
            tvl: item._sum?.tvl ? parseFloat(item._sum.tvl.toString()) : 0,
            percentage: totalTVL > 0 ? ((item._sum?.tvl ? parseFloat(item._sum.tvl.toString()) : 0) / totalTVL) * 100 : 0
        }));

        // Get recent optimization runs for trends
        const recentOptimizationRuns = await prisma.allocation_parameters.findMany({
            take: 20,
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

        const formattedOptimizationData = recentOptimizationRuns.map(run => ({
            date: run.timestamp ? run.timestamp.toISOString() : new Date().toISOString(),
            projected_apy: run.projected_apy ? parseFloat(run.projected_apy.toString()) : 0,
            transaction_costs: run.transaction_costs ? parseFloat(run.transaction_costs.toString()) : 0,
            run_id: run.run_id
        }));

        const chartData = {
            poolAPYDistribution: formattedAPYData,
            tvlByProtocol: formattedTVLData,
            optimizationTrends: formattedOptimizationData
        };

        // Add cache headers
        const response = NextResponse.json(chartData);
        response.headers.set('Cache-Control', 'public, s-maxage=300, stale-while-revalidate=60');

        return response;
    } catch (error) {
        console.error('Dashboard charts API error:', error);
        return NextResponse.json(
            { error: 'Failed to fetch chart data' },
            { status: 500 }
        );
    }
}