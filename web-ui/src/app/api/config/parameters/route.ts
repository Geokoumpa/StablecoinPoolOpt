import { NextRequest, NextResponse } from 'next/server';
import { prisma } from '@/lib/db';

export async function GET(request: NextRequest) {
    try {
        // Get all default configuration parameters
        const parameters = await prisma.default_allocation_parameters.findMany({
            orderBy: {
                parameter_name: 'asc'
            }
        });

        // Group parameters by category for better organization
        const groupedParameters = parameters.reduce((acc: any, param: any) => {
            const category = getParameterCategory(param.parameter_name);
            if (!acc[category]) {
                acc[category] = [];
            }
            acc[category].push(param);
            return acc;
        }, {} as Record<string, any[]>);

        const response = NextResponse.json({
            parameters: groupedParameters,
            rawParameters: parameters
        });

        // Add cache headers
        response.headers.set('Cache-Control', 'public, s-maxage=300, stale-while-revalidate=60');

        return response;
    } catch (error) {
        console.error('Configuration parameters API error:', error);
        return NextResponse.json(
            { error: 'Failed to fetch configuration parameters' },
            { status: 500 }
        );
    }
}

export async function PUT(request: NextRequest) {
    try {
        const body = await request.json();
        const { parameter_name, parameter_value, description } = body;

        if (!parameter_name || parameter_value === undefined) {
            return NextResponse.json(
                { error: 'Parameter name and value are required' },
                { status: 400 }
            );
        }

        // Update or create parameter
        const parameter = await prisma.default_allocation_parameters.upsert({
            where: {
                parameter_name
            },
            update: {
                parameter_value,
                description,
                updated_at: new Date()
            },
            create: {
                parameter_name,
                parameter_value,
                description
            }
        });

        return NextResponse.json(parameter);
    } catch (error) {
        console.error('Update configuration parameter error:', error);
        return NextResponse.json(
            { error: 'Failed to update configuration parameter' },
            { status: 500 }
        );
    }
}

export async function POST(request: NextRequest) {
    try {
        const body = await request.json();
        const { parameter_name, parameter_value, description } = body;

        if (!parameter_name || parameter_value === undefined) {
            return NextResponse.json(
                { error: 'Parameter name and value are required' },
                { status: 400 }
            );
        }

        // Check if parameter already exists
        const existingParameter = await prisma.default_allocation_parameters.findUnique({
            where: {
                parameter_name
            }
        });

        if (existingParameter) {
            return NextResponse.json(
                { error: 'Parameter already exists' },
                { status: 409 }
            );
        }

        // Create new parameter
        const newParameter = await prisma.default_allocation_parameters.create({
            data: {
                parameter_name,
                parameter_value,
                description
            }
        });

        return NextResponse.json(newParameter, { status: 201 });
    } catch (error) {
        console.error('Create configuration parameter error:', error);
        return NextResponse.json(
            { error: 'Failed to create configuration parameter' },
            { status: 500 }
        );
    }
}

export async function DELETE(request: NextRequest) {
    try {
        const { searchParams } = new URL(request.url);
        const parameter_name = searchParams.get('parameter_name');

        if (!parameter_name) {
            return NextResponse.json(
                { error: 'Parameter name is required' },
                { status: 400 }
            );
        }

        // Delete parameter
        const deletedParameter = await prisma.default_allocation_parameters.delete({
            where: {
                parameter_name
            }
        });

        return NextResponse.json({
            message: 'Parameter deleted successfully',
            parameter: deletedParameter
        });
    } catch (error) {
        console.error('Delete configuration parameter error:', error);
        return NextResponse.json(
            { error: 'Failed to delete configuration parameter' },
            { status: 500 }
        );
    }
}

// Helper function to categorize parameters
function getParameterCategory(parameterName: string): string {
    if (parameterName.includes('tvl') || parameterName.includes('alloc')) {
        return 'Allocation Limits';
    }
    if (parameterName.includes('apy') || parameterName.includes('profit')) {
        return 'Profit Optimization';
    }
    if (parameterName.includes('pool') || parameterName.includes('group')) {
        return 'Pool Management';
    }
    if (parameterName.includes('token') || parameterName.includes('marketcap')) {
        return 'Token Management';
    }
    if (parameterName.includes('icebox') || parameterName.includes('recovery')) {
        return 'Risk Management';
    }
    return 'General';
}