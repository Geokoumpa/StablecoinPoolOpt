import { NextRequest, NextResponse } from 'next/server';
import { prisma } from '@/lib/db';

export async function GET(request: NextRequest) {
    try {
        const { searchParams } = new URL(request.url);
        const page = parseInt(searchParams.get('page') || '1');
        const limit = parseInt(searchParams.get('limit') || '50');
        const search = searchParams.get('search');
        const skip = (page - 1) * limit;

        // Build where clause
        const where: any = {
            removed_timestamp: null
        };

        if (search) {
            where.OR = [
                { token_symbol: { contains: search, mode: 'insensitive' } },
                { token_address: { contains: search, mode: 'insensitive' } }
            ];
        }

        // Get total count for pagination
        const totalCount = await prisma.approved_tokens.count({ where });

        // Get approved tokens with pagination
        const tokens = await prisma.approved_tokens.findMany({
            where,
            skip,
            take: limit,
            orderBy: {
                token_symbol: 'asc'
            }
        });

        const response = NextResponse.json({
            tokens,
            pagination: {
                page,
                limit,
                totalCount,
                totalPages: Math.ceil(totalCount / limit)
            }
        });

        // Add cache headers
        response.headers.set('Cache-Control', 'public, s-maxage=300, stale-while-revalidate=60');

        return response;
    } catch (error) {
        console.error('Approved tokens API error:', error);
        return NextResponse.json(
            { error: 'Failed to fetch approved tokens' },
            { status: 500 }
        );
    }
}

export async function POST(request: NextRequest) {
    try {
        const body = await request.json();
        const { token_symbol, token_address } = body;

        // Check if token already exists
        const existingToken = await prisma.approved_tokens.findFirst({
            where: {
                OR: [
                    { token_symbol, removed_timestamp: null },
                    { token_address, removed_timestamp: null }
                ]
            }
        });

        if (existingToken) {
            return NextResponse.json(
                { error: 'Token already exists' },
                { status: 409 }
            );
        }

        // Create new approved token
        const newToken = await prisma.approved_tokens.create({
            data: {
                token_symbol,
                token_address,
                added_timestamp: new Date()
            }
        });

        return NextResponse.json(newToken, { status: 201 });
    } catch (error) {
        console.error('Create approved token error:', error);
        return NextResponse.json(
            { error: 'Failed to create approved token' },
            { status: 500 }
        );
    }
}

export async function DELETE(request: NextRequest) {
    try {
        const { searchParams } = new URL(request.url);
        const token_symbol = searchParams.get('token_symbol');

        if (!token_symbol) {
            return NextResponse.json(
                { error: 'Token symbol is required' },
                { status: 400 }
            );
        }

        // Soft delete by setting removed_timestamp
        const updatedToken = await prisma.approved_tokens.updateMany({
            where: {
                token_symbol,
                removed_timestamp: null
            },
            data: {
                removed_timestamp: new Date()
            }
        });

        if (updatedToken.count === 0) {
            return NextResponse.json(
                { error: 'Token not found' },
                { status: 404 }
            );
        }

        return NextResponse.json({ message: 'Token removed successfully' });
    } catch (error) {
        console.error('Delete approved token error:', error);
        return NextResponse.json(
            { error: 'Failed to remove token' },
            { status: 500 }
        );
    }
}