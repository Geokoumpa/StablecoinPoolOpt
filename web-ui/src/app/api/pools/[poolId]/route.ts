import { NextRequest, NextResponse } from 'next/server';
import { prisma } from '@/lib/db';
import { auth } from '@clerk/nextjs/server';

export async function GET(
    request: NextRequest,
    { params }: { params: { poolId: string } }
) {
    try {
        // Check authentication
        const { userId } = await auth();

        if (!userId) {
            return NextResponse.json(
                { error: 'Unauthorized - Please sign in to access pool data' },
                { status: 401 }
            );
        }

        const { poolId } = params;

        // Get specific pool
        const pool = await prisma.pools.findUnique({
            where: {
                pool_id: poolId
            },
            include: {
                pool_daily_metrics: {
                    take: 30, // Last 30 days of metrics
                    orderBy: {
                        date: 'desc'
                    }
                }
            }
        });

        if (!pool) {
            return NextResponse.json(
                { error: 'Pool not found' },
                { status: 404 }
            );
        }

        const response = NextResponse.json(pool);

        // Add cache headers
        response.headers.set('Cache-Control', 'public, s-maxage=300, stale-while-revalidate=60');

        return response;
    } catch (error) {
        console.error('Pool detail API error:', error);
        return NextResponse.json(
            { error: 'Failed to fetch pool details' },
            { status: 500 }
        );
    }
}

export async function PUT(
    request: NextRequest,
    { params }: { params: { poolId: string } }
) {
    try {
        // Check authentication
        const { userId } = await auth();

        if (!userId) {
            return NextResponse.json(
                { error: 'Unauthorized - Please sign in to access pool data' },
                { status: 401 }
            );
        }

        const { poolId } = params;
        const body = await request.json();

        // Update the pool
        const updatedPool = await prisma.pools.update({
            where: {
                pool_id: poolId
            },
            data: {
                ...body,
                last_updated: new Date()
            }
        });

        return NextResponse.json(updatedPool);
    } catch (error) {
        console.error('Update pool error:', error);
        return NextResponse.json(
            { error: 'Failed to update pool' },
            { status: 500 }
        );
    }
}