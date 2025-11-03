"use client"

import { Bar, BarChart, ResponsiveContainer, XAxis, YAxis } from "recharts"
import {
    ChartContainer,
    ChartTooltip,
    ChartTooltipContent,
} from "@/components/ui/chart"

interface PoolAPYDistributionProps {
    data: Array<{
        name: string
        apy: number
    }>
}

const chartConfig = {
    apy: {
        label: "APY (%)",
        color: "hsl(var(--chart-1))",
    },
}

export function PoolAPYDistribution({ data }: PoolAPYDistributionProps) {
    return (
        <ChartContainer config={chartConfig} className="h-[300px] w-full">
            <BarChart data={data}>
                <XAxis
                    dataKey="name"
                    tick={{ fontSize: 12 }}
                    angle={-45}
                    textAnchor="end"
                    height={60}
                />
                <YAxis
                    tick={{ fontSize: 12 }}
                    label={{ value: 'APY (%)', angle: -90, position: 'insideLeft' }}
                />
                <ChartTooltip
                    cursor={false}
                    content={<ChartTooltipContent />}
                />
                <Bar dataKey="apy" fill="var(--color-apy)" radius={[4, 4, 0, 0]} />
            </BarChart>
        </ChartContainer>
    )
}