import { NextResponse } from 'next/server';

// Common error response handler
export function handleApiError(
    error: any,
    message: string,
    statusCode: number = 500
) {
    console.error(`${message}:`, error);
    return NextResponse.json(
        { error: message },
        { status: statusCode }
    );
}

// Success response with caching
export function createSuccessResponse(
    data: any,
    cacheTime: number = 300,
    staleTime: number = 60
) {
    const response = NextResponse.json(data);
    response.headers.set('Cache-Control', `public, s-maxage=${cacheTime}, stale-while-revalidate=${staleTime}`);
    return response;
}

// Validation error response
export function createValidationError(message: string, field?: string) {
    return NextResponse.json(
        {
            error: 'Validation Error',
            message,
            field
        },
        { status: 400 }
    );
}

// Not found error response
export function createNotFoundError(resource: string) {
    return NextResponse.json(
        { error: `${resource} not found` },
        { status: 404 }
    );
}

// Conflict error response
export function createConflictError(message: string) {
    return NextResponse.json(
        { error: message },
        { status: 409 }
    );
}

// Unauthorized error response
export function createUnauthorizedError(message: string = 'Unauthorized') {
    return NextResponse.json(
        { error: message },
        { status: 401 }
    );
}

// Pagination helper
export function createPaginationResponse(
    data: any[],
    page: number,
    limit: number,
    totalCount: number
) {
    return {
        data,
        pagination: {
            page,
            limit,
            totalCount,
            totalPages: Math.ceil(totalCount / limit),
            hasNextPage: page < Math.ceil(totalCount / limit),
            hasPreviousPage: page > 1
        }
    };
}

// Query parameter helper
export function getQueryParams(request: Request) {
    const { searchParams } = new URL(request.url);
    return {
        page: parseInt(searchParams.get('page') || '1'),
        limit: parseInt(searchParams.get('limit') || '10'),
        search: searchParams.get('search') || '',
        protocol: searchParams.get('protocol') || '',
        chain: searchParams.get('chain') || '',
        days: parseInt(searchParams.get('days') || '30'),
        sortBy: searchParams.get('sortBy') || 'createdAt',
        sortOrder: searchParams.get('sortOrder') || 'desc'
    };
}

// Date range helper
export function getDateRange(days: number) {
    const endDate = new Date();
    const startDate = new Date();
    startDate.setDate(endDate.getDate() - days);
    return { startDate, endDate };
}

// Validate required fields
export function validateRequiredFields(
    body: any,
    requiredFields: string[]
): { isValid: boolean; missingFields: string[] } {
    const missingFields = requiredFields.filter(field => !body[field]);
    return {
        isValid: missingFields.length === 0,
        missingFields
    };
}

// Sanitize search term
export function sanitizeSearchTerm(search: string): string {
    return search.trim().replace(/[^a-zA-Z0-9\s\-_]/g, '');
}

// Build where clause for search
export function buildSearchWhereClause(search: string, searchFields: string[]) {
    if (!search) return {};

    const sanitizedSearch = sanitizeSearchTerm(search);
    return {
        OR: searchFields.map(field => ({
            [field]: {
                contains: sanitizedSearch,
                mode: 'insensitive' as const
            }
        }))
    };
}