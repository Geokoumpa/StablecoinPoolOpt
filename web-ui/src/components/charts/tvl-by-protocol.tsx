"use client"

import { Pie, PieChart, ResponsiveContainer, Cell, Legend } from "recharts"
import {
    ChartContainer,
    ChartTooltip,
    ChartTooltipContent,
} from "@/components/ui/chart"

interface TVLByProtocolProps {
    data: Array<{
        name: string
        tvl: number
        percentage?: number
    }>
}

const COLORS = [
    "hsl(var(--chart-1))",
    "hsl(var(--chart-2))",
    "hsl(var(--chart-3))",
    "hsl(var(--chart-4))",
    "hsl(var(--chart-5))",
    "hsl(var(--chart-6))",
]

const chartConfig = {
    tvl: {
        label: "TVL ($)",
    },
}

export function TVLByProtocol({ data }: TVLByProtocolProps) {
    const formattedData = data.map((item, index) => ({
        ...item,
        fill: COLORS[index % COLORS.length],
    }))

    return (
        <ChartContainer config={chartConfig} className="h-[300px] w-full">
            <PieChart>
                <Pie
                    data={formattedData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percentage }) =>
                        percentage ? `${name}: ${percentage.toFixed(1)}%` : name
                    }
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="tvl"
                >
                    {formattedData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.fill} />
                    ))}
                </Pie>
                <ChartTooltip
                    content={<ChartTooltipContent />}
                    formatter={(value: number) => [`$${value.toLocaleString()}`, 'TVL']}
                />
                <Legend
                    verticalAlign="bottom"
                    height={36}
                    formatter={(value, entry: any) =>
                        `${value}: $${entry.payload.tvl.toLocaleString()}`
                    }
                />
            </PieChart>
        </ChartContainer>
    )
}