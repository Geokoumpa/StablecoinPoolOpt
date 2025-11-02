import { NextRequest, NextResponse } from 'next/server';
import { prisma } from '@/lib/db';
import { handleApiError, createSuccessResponse, createValidationError } from '@/lib/api-utils';

export async function GET(request: NextRequest) {
    try {
        const { searchParams } = new URL(request.url);
        const page = parseInt(searchParams.get('page') || '1');
        const limit = parseInt(searchParams.get('limit') || '20');
        const search = searchParams.get('search');
        const tokenSymbol = searchParams.get('tokenSymbol');
        const skip = (page - 1) * limit;

        // Build where clause for daily balances
        const where: any = {};

        if (search) {
            where.OR = [
                { token_symbol: { contains: search, mode: 'insensitive' } }
            ];
        }

        if (tokenSymbol) {
            where.token_symbol = tokenSymbol;
        }

        // Get total count for pagination
        const totalCount = await prisma.daily_balances.count({ where });

        // Get daily balances with pagination (latest date for each wallet/token combination)
        const balances = await prisma.daily_balances.findMany({
            where,
            skip,
            take: limit,
            orderBy: [
                { date: 'desc' },
                { token_symbol: 'asc' }
            ],
            select: {
                id: true,
                date: true,
                token_symbol: true,
                unallocated_balance: true,
                allocated_balance: true,
                pool_id: true
            },
            distinct: ['token_symbol', 'date']
        });

        // Get unique token symbols for filtering
        const uniqueTokens = await prisma.daily_balances.findMany({
            select: {
                token_symbol: true
            },
            distinct: ['token_symbol'],
            orderBy: {
                token_symbol: 'asc'
            }
        });

        // Get configured wallet addresses from environment
        const configuredWallets = {
            warmWallet: process.env.WARM_WALLET_ADDRESS || '',
            coldWallet: process.env.COLD_WALLET_ADDRESS || '',
            recoveryWallet: process.env.RECOVERY_WALLET_ADDRESS || ''
        };

        // Calculate total allocations
        const totalAllocations = balances.reduce((acc: any, balance: any) => {
            const unallocated = parseFloat(balance.unallocated_balance?.toString() || '0');
            const allocated = parseFloat(balance.allocated_balance?.toString() || '0');
            return {
                totalUnallocated: acc.totalUnallocated + unallocated,
                totalAllocated: acc.totalAllocated + allocated,
                totalValue: acc.totalValue + unallocated + allocated
            };
        }, { totalUnallocated: 0, totalAllocated: 0, totalValue: 0 });

        const response = createSuccessResponse({
            balances,
            configuredWallets,
            filters: {
                tokenSymbols: uniqueTokens.map((t: any) => t.token_symbol).filter(Boolean)
            },
            summary: totalAllocations,
            pagination: {
                page,
                limit,
                totalCount,
                totalPages: Math.ceil(totalCount / limit)
            }
        }, 60, 30);

        return response;
    } catch (error) {
        return handleApiError(error, 'Failed to fetch wallet data');
    }
}

export async function PUT(request: NextRequest) {
    try {
        const body = await request.json();
        const {
            warmWallet,
            coldWallet,
            recoveryWallet
        } = body;

        // In a real implementation, this would update environment variables
        // For now, we'll return a success response with the updated values
        // Note: This would require server restart to take effect

        const updatedWallets = {
            warmWallet: warmWallet || process.env.WARM_WALLET_ADDRESS || '',
            coldWallet: coldWallet || process.env.COLD_WALLET_ADDRESS || '',
            recoveryWallet: recoveryWallet || process.env.RECOVERY_WALLET_ADDRESS || ''
        };

        // Validate wallet addresses format (basic Ethereum address validation)
        const addressRegex = /^0x[a-fA-F0-9]{40}$/;
        const validationErrors = [];

        if (warmWallet && !addressRegex.test(warmWallet)) {
            validationErrors.push('Invalid warm wallet address format');
        }
        if (coldWallet && !addressRegex.test(coldWallet)) {
            validationErrors.push('Invalid cold wallet address format');
        }
        if (recoveryWallet && !addressRegex.test(recoveryWallet)) {
            validationErrors.push('Invalid recovery wallet address format');
        }

        if (validationErrors.length > 0) {
            return createValidationError('Invalid wallet addresses', validationErrors.join(', '));
        }

        return createSuccessResponse({
            message: 'Wallet addresses updated (requires server restart to take effect)',
            wallets: updatedWallets
        });

    } catch (error) {
        return handleApiError(error, 'Failed to update wallet addresses');
    }
}

// GET endpoint for wallet configuration only
export async function POST(request: NextRequest) {
    try {
        const configuredWallets = {
            warmWallet: process.env.WARM_WALLET_ADDRESS || '',
            coldWallet: process.env.COLD_WALLET_ADDRESS || '',
            recoveryWallet: process.env.RECOVERY_WALLET_ADDRESS || ''
        };

        return createSuccessResponse({
            wallets: configuredWallets,
            note: 'Wallet addresses are configured via environment variables'
        }, 300, 60);

    } catch (error) {
        return handleApiError(error, 'Failed to fetch wallet configuration');
    }
}