import React from "react"
import { z } from "zod"

// Form validation schemas using Zod based on actual database schema
export const poolConfigurationSchema = z.object({
    name: z.string().min(1, "Pool name is required"),
    protocol: z.string().min(1, "Protocol is required"),
    address: z.string().min(1, "Address is required").regex(/^0x[a-fA-F0-9]{40}$/, "Invalid Ethereum address"),
    minAllocation: z.number().min(0, "Minimum allocation must be positive"),
    maxAllocation: z.number().min(0, "Maximum allocation must be positive"),
}).refine((data: any) => data.maxAllocation >= data.minAllocation, {
    message: "Maximum allocation must be greater than or equal to minimum allocation",
    path: ["maxAllocation"],
})

export const allocationParametersSchema = z.object({
    tvl_limit_percentage: z.number().min(0, "TVL limit percentage must be positive").max(1, "TVL limit percentage cannot exceed 100%"),
    max_alloc_percentage: z.number().min(0, "Max allocation percentage must be positive").max(1, "Max allocation percentage cannot exceed 100%"),
    conversion_rate: z.number().min(0, "Conversion rate must be positive"),
    min_pools: z.number().int().min(1, "Minimum pools must be at least 1"),
    profit_optimization: z.boolean(),
    token_marketcap_limit: z.number().min(0, "Token market cap limit must be positive"),
    pool_tvl_limit: z.number().min(0, "Pool TVL limit must be positive"),
    pool_apy_limit: z.number().min(0, "Pool APY limit must be positive"),
    pool_pair_tvl_ratio_min: z.number().min(0, "Pool pair TVL ratio min must be positive").max(1, "Cannot exceed 100%"),
    pool_pair_tvl_ratio_max: z.number().min(0, "Pool pair TVL ratio max must be positive").max(1, "Cannot exceed 100%"),
    group1_max_pct: z.number().min(0, "Group 1 max percentage must be positive").max(1, "Cannot exceed 100%"),
    group2_max_pct: z.number().min(0, "Group 2 max percentage must be positive").max(1, "Cannot exceed 100%"),
    group3_max_pct: z.number().min(0, "Group 3 max percentage must be positive").max(1, "Cannot exceed 100%"),
    position_max_pct_total_assets: z.number().min(0, "Position max percentage of total assets must be positive").max(1, "Cannot exceed 100%"),
    position_max_pct_pool_tvl: z.number().min(0, "Position max percentage of pool TVL must be positive").max(1, "Cannot exceed 100%"),
    group1_apy_delta_max: z.number().min(0, "Group 1 APY delta max must be positive"),
    group1_7d_stddev_max: z.number().min(0, "Group 1 7-day stddev max must be positive"),
    group1_30d_stddev_max: z.number().min(0, "Group 1 30-day stddev max must be positive"),
    group2_apy_delta_max: z.number().min(0, "Group 2 APY delta max must be positive"),
    group2_7d_stddev_max: z.number().min(0, "Group 2 7-day stddev max must be positive"),
    group2_30d_stddev_max: z.number().min(0, "Group 2 30-day stddev max must be positive"),
    group3_apy_delta_min: z.number().min(0, "Group 3 APY delta min must be positive"),
    group3_7d_stddev_min: z.number().min(0, "Group 3 7-day stddev min must be positive"),
    group3_30d_stddev_min: z.number().min(0, "Group 3 30-day stddev min must be positive"),
    icebox_ohlc_l_threshold_pct: z.number().min(0, "Icebox OHLC L threshold percentage must be positive").max(1, "Cannot exceed 100%"),
    icebox_ohlc_l_days_threshold: z.number().int().min(1, "Icebox OHLC L days threshold must be at least 1"),
    icebox_ohlc_c_threshold_pct: z.number().min(0, "Icebox OHLC C threshold percentage must be positive").max(1, "Cannot exceed 100%"),
    icebox_ohlc_c_days_threshold: z.number().int().min(1, "Icebox OHLC C days threshold must be at least 1"),
    icebox_recovery_l_days_threshold: z.number().int().min(1, "Icebox recovery L days threshold must be at least 1"),
    icebox_recovery_c_days_threshold: z.number().int().min(1, "Icebox recovery C days threshold must be at least 1"),
})

export const protocolSchema = z.object({
    name: z.string().min(1, "Protocol name is required"),
    description: z.string().min(1, "Description is required"),
    website: z.string().url("Please enter a valid URL"),
    isActive: z.boolean().default(true),
})

export const tokenSchema = z.object({
    symbol: z.string().min(1, "Token symbol is required").max(10, "Symbol must be 10 characters or less"),
    name: z.string().min(1, "Token name is required"),
    address: z.string().min(1, "Address is required").regex(/^0x[a-fA-F0-9]{40}$/, "Invalid Ethereum address"),
    decimals: z.number().int().min(0, "Decimals must be a non-negative integer").max(18, "Decimals cannot exceed 18"),
    isStablecoin: z.boolean().default(false),
    isActive: z.boolean().default(true),
})

export const walletAddressSchema = z.object({
    address: z.string().min(1, "Wallet address is required").regex(/^0x[a-fA-F0-9]{40}$/, "Invalid Ethereum address"),
    label: z.string().min(1, "Label is required"),
    description: z.string().optional(),
})

// Type exports
export type PoolConfigurationFormData = z.infer<typeof poolConfigurationSchema>
export type AllocationParametersFormData = z.infer<typeof allocationParametersSchema>
export type ProtocolFormData = z.infer<typeof protocolSchema>
export type TokenFormData = z.infer<typeof tokenSchema>
export type WalletAddressFormData = z.infer<typeof walletAddressSchema>

// Validation utility functions
export const validateEmail = (email: string): boolean => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
    return emailRegex.test(email)
}

export const validateEthereumAddress = (address: string): boolean => {
    const addressRegex = /^0x[a-fA-F0-9]{40}$/
    return addressRegex.test(address)
}

export const validateRequired = (value: string): boolean => {
    return value.trim().length > 0
}

export const validateMinLength = (value: string, minLength: number): boolean => {
    return value.length >= minLength
}

export const validateMaxLength = (value: string, maxLength: number): boolean => {
    return value.length <= maxLength
}

export const validateNumberRange = (value: number, min: number, max: number): boolean => {
    return value >= min && value <= max
}

export const validatePositiveNumber = (value: number): boolean => {
    return value > 0
}

export const validateNonNegativeNumber = (value: number): boolean => {
    return value >= 0
}

export const validateUrl = (url: string): boolean => {
    try {
        new URL(url)
        return true
    } catch {
        return false
    }
}

// Form validation helper
export const validateForm = <T>(schema: z.ZodSchema<T>, data: unknown): { success: boolean; errors: Record<string, string>; data?: T } => {
    const result = schema.safeParse(data)

    if (!result.success) {
        const errors: Record<string, string> = {}
        result.error.issues.forEach((issue: any) => {
            const path = issue.path.join('.')
            errors[path] = issue.message
        })
        return { success: false, errors }
    }

    return { success: true, errors: {}, data: result.data }
}

// Real-time validation hook simulation (for client-side validation)
export const createFieldValidator = <T extends z.ZodObject<any>>(schema: T) => {
    return (field: keyof z.infer<T>, value: unknown): string | null => {
        try {
            const fieldSchema = (schema as any).shape[field as string] as z.ZodSchema<any>
            fieldSchema.parse(value)
            return null
        } catch (error) {
            if (error instanceof z.ZodError) {
                return error.issues[0]?.message || 'Invalid value'
            }
            return 'Invalid value'
        }
    }
}

// Custom validation hooks for React components
export const useFormValidation = <T extends z.ZodObject<any>>(schema: T) => {
    const [errors, setErrors] = React.useState<Record<string, string>>({})

    const validateField = React.useCallback((field: keyof z.infer<T>, value: unknown) => {
        const validator = createFieldValidator(schema)
        const error = validator(field, value)

        setErrors(prev => ({
            ...prev,
            [field as string]: error || ''
        }))

        return !error
    }, [schema])

    const validateForm = React.useCallback((data: unknown): boolean => {
        const result = schema.safeParse(data)

        if (!result.success) {
            const newErrors: Record<string, string> = {}
            result.error.issues.forEach((issue: any) => {
                const path = issue.path.join('.')
                newErrors[path] = issue.message
            })
            setErrors(newErrors)
            return false
        }

        setErrors({})
        return true
    }, [schema])

    const clearErrors = React.useCallback(() => {
        setErrors({})
    }, [])

    const clearFieldError = React.useCallback((field: keyof z.infer<T>) => {
        setErrors(prev => ({
            ...prev,
            [field as string]: ''
        }))
    }, [])

    return {
        errors,
        validateField,
        validateForm,
        clearErrors,
        clearFieldError,
        hasErrors: Object.values(errors).some(error => error.length > 0)
    }
}