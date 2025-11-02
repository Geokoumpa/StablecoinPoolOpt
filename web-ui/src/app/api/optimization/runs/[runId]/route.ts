import { NextRequest, NextResponse } from 'next/server';
import { prisma } from '@/lib/db';

export async function GET(
    request: NextRequest,
    { params }: { params: { runId: string } }
) {
    try {
        const { runId } = params;

        // Get specific optimization run
        const run = await prisma.allocation_parameters.findUnique({
            where: {
                run_id: runId
            }
        });

        if (!run) {
            return NextResponse.json(
                { error: 'Optimization run not found' },
                { status: 404 }
            );
        }

        // Get asset allocations for this run
        const assetAllocations = await prisma.asset_allocations.findMany({
            where: {
                run_id: runId
            },
            orderBy: {
                step_number: 'asc'
            }
        });

        const response = NextResponse.json({
            run,
            assetAllocations
        });

        // Add cache headers
        response.headers.set('Cache-Control', 'public, s-maxage=300, stale-while-revalidate=60');

        return response;
    } catch (error) {
        console.error('Optimization run detail API error:', error);
        return NextResponse.json(
            { error: 'Failed to fetch optimization run details' },
            { status: 500 }
        );
    }
}

export async function PUT(
    request: NextRequest,
    { params }: { params: { runId: string } }
) {
    try {
        const { runId } = params;
        const body = await request.json();

        // Update optimization run
        const updatedRun = await prisma.allocation_parameters.update({
            where: {
                run_id: runId
            },
            data: {
                ...body,
                updated_at: new Date()
            }
        });

        return NextResponse.json(updatedRun);
    } catch (error) {
        console.error('Update optimization run error:', error);
        return NextResponse.json(
            { error: 'Failed to update optimization run' },
            { status: 500 }
        );
    }
}

export async function DELETE(
    request: NextRequest,
    { params }: { params: { runId: string } }
) {
    try {
        const { runId } = params;

        // Delete asset allocations first (foreign key constraint)
        await prisma.asset_allocations.deleteMany({
            where: {
                run_id: runId
            }
        });

        // Delete optimization run
        await prisma.allocation_parameters.delete({
            where: {
                run_id: runId
            }
        });

        return NextResponse.json({ message: 'Optimization run deleted successfully' });
    } catch (error) {
        console.error('Delete optimization run error:', error);
        return NextResponse.json(
            { error: 'Failed to delete optimization run' },
            { status: 500 }
        );
    }
}