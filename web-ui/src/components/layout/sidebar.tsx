'use client';

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
    HomeIcon,
    BarChart3Icon,
    DatabaseIcon,
    SettingsIcon,
    UsersIcon,
    TrendingUpIcon
} from "lucide-react";

const navigation = [
    {
        name: "Dashboard",
        href: "/",
        icon: HomeIcon,
    },
    {
        name: "Optimization",
        href: "/optimization",
        icon: TrendingUpIcon,
    },
    {
        name: "Pools",
        href: "/pools",
        icon: DatabaseIcon,
    },
    {
        name: "Protocols",
        href: "/protocols",
        icon: UsersIcon,
    },
    {
        name: "Configuration",
        href: "/config",
        icon: SettingsIcon,
    },
];

export function Sidebar() {
    const pathname = usePathname();

    return (
        <div className="flex h-full w-64 flex-col bg-gray-800">
            <div className="flex h-16 shrink-0 items-center justify-center bg-blue-600">
                <span className="text-white font-bold">SPO</span>
            </div>
            <nav className="flex-1 px-2 space-y-1">
                {navigation.map((item) => {
                    const isActive = pathname === item.href;

                    return (
                        <Link
                            key={item.name}
                            href={item.href}
                            className={`group flex items-center px-2 py-2 text-sm font-medium rounded-md transition-colors ${isActive
                                ? "bg-gray-900 text-white"
                                : "text-gray-300 hover:bg-gray-700 hover:text-white"
                                }`}
                        >
                            <item.icon
                                className={`mr-3 h-5 w-5 ${isActive ? "text-blue-400" : "text-gray-400 group-hover:text-blue-300"
                                    }`}
                            />
                            {item.name}
                        </Link>
                    );
                })}
            </nav>
        </div>
    );
}