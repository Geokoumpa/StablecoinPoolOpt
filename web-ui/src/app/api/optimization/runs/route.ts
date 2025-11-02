import { NextRequest, NextResponse } from 'next/server';
import { prisma } from '@/lib/db';
import { auth } from '@clerk/nextjs/server';

export async function GET(request: NextRequest) {
    try {
        // Check authentication
        const { userId } = await auth();

        if (!userId) {
            return NextResponse.json(
                { error: 'Unauthorized - Please sign in to access optimization runs' },
                { status: 401 }
            );
        }
        const { searchParams } = new URL(request.url);
        const page = parseInt(searchParams.get('page') || '1');
        const limit = parseInt(searchParams.get('limit') || '10');
        const skip = (page - 1) * limit;

        // Get total count for pagination
        const totalCount = await prisma.allocation_parameters.count();

        // Get optimization runs with pagination
        const runs = await prisma.allocation_parameters.findMany({
            skip,
            take: limit,
            orderBy: {
                timestamp: 'desc'
            },
            select: {
                run_id: true,
                timestamp: true,
                projected_apy: true,
                transaction_costs: true,
                tvl_limit_percentage: true,
                max_alloc_percentage: true,
                min_pools: true,
                profit_optimization: true,
                pool_tvl_limit: true,
                pool_apy_limit: true
            }
        });

        const response = NextResponse.json({
            runs,
            pagination: {
                page,
                limit,
                totalCount,
                totalPages: Math.ceil(totalCount / limit)
            }
        });

        // Add cache headers
        response.headers.set('Cache-Control', 'public, s-maxage=30, stale-while-revalidate=15');

        return response;
    } catch (error) {
        console.error('Optimization runs API error:', error);
        return NextResponse.json(
            { error: 'Failed to fetch optimization runs' },
            { status: 500 }
        );
    }
}

export async function POST(request: NextRequest) {
    try {
        // Check authentication
        const { userId } = await auth();

        if (!userId) {
            return NextResponse.json(
                { error: 'Unauthorized - Please sign in to create optimization runs' },
                { status: 401 }
            );
        }
        const body = await request.json();

        // Create a new optimization run
        const newRun = await prisma.allocation_parameters.create({
            data: {
                run_id: crypto.randomUUID(),
                timestamp: new Date(),
                ...body
            }
        });

        return NextResponse.json(newRun, { status: 201 });
    } catch (error) {
        console.error('Create optimization run error:', error);
        return NextResponse.json(
            { error: 'Failed to create optimization run' },
            { status: 500 }
        );
    }
}