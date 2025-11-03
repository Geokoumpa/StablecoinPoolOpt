"use client"

import { Line, LineChart, ResponsiveContainer, XAxis, YAxis } from "recharts"
import {
    ChartContainer,
    ChartTooltip,
    ChartTooltipContent,
} from "@/components/ui/chart"

interface OptimizationTrendsProps {
    data: Array<{
        date: string
        projected_apy: number
        transaction_costs: number
        run_id: string
    }>
}

const chartConfig = {
    projected_apy: {
        label: "Projected APY (%)",
        color: "hsl(var(--chart-1))",
    },
    transaction_costs: {
        label: "Transaction Costs ($)",
        color: "hsl(var(--chart-2))",
    },
}

export function OptimizationTrends({ data }: OptimizationTrendsProps) {
    const formattedData = data.map(item => ({
        ...item,
        date: new Date(item.date).toLocaleDateString(),
    })).reverse() // Show oldest to newest

    return (
        <ChartContainer config={chartConfig} className="h-[300px] w-full">
            <LineChart data={formattedData}>
                <XAxis
                    dataKey="date"
                    tick={{ fontSize: 12 }}
                    angle={-45}
                    textAnchor="end"
                    height={60}
                />
                <YAxis
                    yAxisId="left"
                    tick={{ fontSize: 12 }}
                    label={{ value: 'Projected APY (%)', angle: -90, position: 'insideLeft' }}
                />
                <YAxis
                    yAxisId="right"
                    orientation="right"
                    tick={{ fontSize: 12 }}
                    label={{ value: 'Transaction Costs ($)', angle: 90, position: 'insideRight' }}
                />
                <ChartTooltip
                    content={<ChartTooltipContent />}
                    labelFormatter={(label) => `Date: ${label}`}
                />
                <Line
                    yAxisId="left"
                    type="monotone"
                    dataKey="projected_apy"
                    stroke="var(--color-projected_apy)"
                    strokeWidth={2}
                    dot={{ fill: "var(--color-projected_apy)", strokeWidth: 2, r: 4 }}
                    activeDot={{ r: 6 }}
                />
                <Line
                    yAxisId="right"
                    type="monotone"
                    dataKey="transaction_costs"
                    stroke="var(--color-transaction_costs)"
                    strokeWidth={2}
                    dot={{ fill: "var(--color-transaction_costs)", strokeWidth: 2, r: 4 }}
                    activeDot={{ r: 6 }}
                />
            </LineChart>
        </ChartContainer>
    )
}