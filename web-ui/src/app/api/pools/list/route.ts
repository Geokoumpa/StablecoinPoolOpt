import { NextRequest, NextResponse } from 'next/server';
import { prisma } from '@/lib/db';
import { auth } from '@clerk/nextjs/server';

export async function GET(request: NextRequest) {
    try {
        // Check authentication
        const { userId } = await auth();

        if (!userId) {
            return NextResponse.json(
                { error: 'Unauthorized - Please sign in to access pools data' },
                { status: 401 }
            );
        }
        const { searchParams } = new URL(request.url);
        const page = parseInt(searchParams.get('page') || '1');
        const limit = parseInt(searchParams.get('limit') || '20');
        const protocol = searchParams.get('protocol');
        const chain = searchParams.get('chain');
        const search = searchParams.get('search');
        const skip = (page - 1) * limit;

        // Build where clause
        const where: any = {
            currently_filtered_out: false
        };

        if (protocol) {
            where.protocol = protocol;
        }

        if (chain) {
            where.chain = chain;
        }

        if (search) {
            where.OR = [
                { name: { contains: search, mode: 'insensitive' } },
                { symbol: { contains: search, mode: 'insensitive' } },
                { protocol: { contains: search, mode: 'insensitive' } }
            ];
        }

        // Get total count for pagination
        const totalCount = await prisma.pools.count({ where });

        // Get pools with pagination
        const pools = await prisma.pools.findMany({
            where,
            skip,
            take: limit,
            orderBy: [
                { tvl: 'desc' },
                { apy: 'desc' }
            ],
            select: {
                pool_id: true,
                name: true,
                chain: true,
                protocol: true,
                symbol: true,
                tvl: true,
                apy: true,
                last_updated: true,
                pool_address: true,
                underlying_tokens: true,
                underlying_token_addresses: true
            }
        });

        // Get unique protocols and chains for filtering
        const [protocols, chains] = await Promise.all([
            prisma.pools.findMany({
                where: { currently_filtered_out: false },
                select: { protocol: true },
                distinct: ['protocol']
            }),
            prisma.pools.findMany({
                where: { currently_filtered_out: false },
                select: { chain: true },
                distinct: ['chain']
            })
        ]);

        const response = NextResponse.json({
            pools,
            filters: {
                protocols: protocols.map((p: any) => p.protocol).filter(Boolean),
                chains: chains.map((c: any) => c.chain).filter(Boolean)
            },
            pagination: {
                page,
                limit,
                totalCount,
                totalPages: Math.ceil(totalCount / limit)
            }
        });

        // Add cache headers
        response.headers.set('Cache-Control', 'public, s-maxage=60, stale-while-revalidate=30');

        return response;
    } catch (error) {
        console.error('Pools list API error:', error);
        return NextResponse.json(
            { error: 'Failed to fetch pools list' },
            { status: 500 }
        );
    }
}