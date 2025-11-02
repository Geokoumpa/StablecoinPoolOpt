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
            where.protocol_name = {
                contains: search,
                mode: 'insensitive'
            };
        }

        // Get total count for pagination
        const totalCount = await prisma.approved_protocols.count({ where });

        // Get approved protocols with pagination
        const protocols = await prisma.approved_protocols.findMany({
            where,
            skip,
            take: limit,
            orderBy: {
                protocol_name: 'asc'
            }
        });

        const response = NextResponse.json({
            protocols,
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
        console.error('Approved protocols API error:', error);
        return NextResponse.json(
            { error: 'Failed to fetch approved protocols' },
            { status: 500 }
        );
    }
}

export async function POST(request: NextRequest) {
    try {
        const body = await request.json();
        const { protocol_name } = body;

        // Check if protocol already exists
        const existingProtocol = await prisma.approved_protocols.findFirst({
            where: {
                protocol_name,
                removed_timestamp: null
            }
        });

        if (existingProtocol) {
            return NextResponse.json(
                { error: 'Protocol already exists' },
                { status: 409 }
            );
        }

        // Create new approved protocol
        const newProtocol = await prisma.approved_protocols.create({
            data: {
                protocol_name,
                added_timestamp: new Date()
            }
        });

        return NextResponse.json(newProtocol, { status: 201 });
    } catch (error) {
        console.error('Create approved protocol error:', error);
        return NextResponse.json(
            { error: 'Failed to create approved protocol' },
            { status: 500 }
        );
    }
}

export async function DELETE(request: NextRequest) {
    try {
        const { searchParams } = new URL(request.url);
        const protocol_name = searchParams.get('protocol_name');

        if (!protocol_name) {
            return NextResponse.json(
                { error: 'Protocol name is required' },
                { status: 400 }
            );
        }

        // Soft delete by setting removed_timestamp
        const updatedProtocol = await prisma.approved_protocols.updateMany({
            where: {
                protocol_name,
                removed_timestamp: null
            },
            data: {
                removed_timestamp: new Date()
            }
        });

        if (updatedProtocol.count === 0) {
            return NextResponse.json(
                { error: 'Protocol not found' },
                { status: 404 }
            );
        }

        return NextResponse.json({ message: 'Protocol removed successfully' });
    } catch (error) {
        console.error('Delete approved protocol error:', error);
        return NextResponse.json(
            { error: 'Failed to remove protocol' },
            { status: 500 }
        );
    }
}