import { useUser, SignOutButton } from "@clerk/clerk-react";
import { Link, useLocation } from "react-router";
import {
    HomeIcon,
    ChartBarIcon,
    ServerIcon,
    CircleStackIcon,
    Cog6ToothIcon,
    ArrowRightOnRectangleIcon
} from "@heroicons/react/24/outline";

interface DashboardLayoutProps {
    children: React.ReactNode;
}

export function DashboardLayout({ children }: DashboardLayoutProps) {
    const { user } = useUser();
    const location = useLocation();

    const navigation = [
        { name: "Dashboard", href: "/", icon: HomeIcon },
        { name: "Optimization", href: "/optimization", icon: ChartBarIcon },
        { name: "Pools", href: "/pools", icon: CircleStackIcon },
        { name: "Protocols", href: "/protocols", icon: ServerIcon },
        { name: "Configuration", href: "/config", icon: Cog6ToothIcon },
    ];

    const isActive = (href: string) => {
        if (href === "/") {
            return location.pathname === "/";
        }
        return location.pathname.startsWith(href);
    };

    return (
        <div className="min-h-screen bg-gray-50">
            {/* Sidebar */}
            <div className="fixed inset-y-0 left-0 z-50 w-64 bg-white shadow-lg transform transition-transform duration-300 ease-in-out lg:translate-x-0 lg:static lg:inset-0">
                <div className="flex items-center justify-center h-16 px-4 bg-blue-600">
                    <h1 className="text-white text-xl font-bold">Stablecoin Pool Opt</h1>
                </div>

                <nav className="mt-8">
                    <div className="px-4 space-y-2">
                        {navigation.map((item) => {
                            const Icon = item.icon;
                            return (
                                <Link
                                    key={item.name}
                                    to={item.href}
                                    className={`group flex items-center px-3 py-2 text-sm font-medium rounded-md transition-colors ${isActive(item.href)
                                            ? "bg-blue-50 text-blue-700 border-r-2 border-blue-700"
                                            : "text-gray-600 hover:bg-gray-50 hover:text-gray-900"
                                        }`}
                                >
                                    <Icon
                                        className={`mr-3 h-5 w-5 flex-shrink-0 ${isActive(item.href) ? "text-blue-500" : "text-gray-400 group-hover:text-gray-500"
                                            }`}
                                    />
                                    {item.name}
                                </Link>
                            );
                        })}
                    </div>
                </nav>

                {/* User section */}
                <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-gray-200">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center">
                            <div className="h-8 w-8 rounded-full bg-blue-100 flex items-center justify-center">
                                <span className="text-sm font-medium text-blue-600">
                                    {user?.firstName?.charAt(0) || user?.emailAddresses[0]?.emailAddress?.charAt(0) || "U"}
                                </span>
                            </div>
                            <div className="ml-3">
                                <p className="text-sm font-medium text-gray-900">
                                    {user?.firstName || user?.emailAddresses[0]?.emailAddress || "User"}
                                </p>
                                <p className="text-xs text-gray-500">Admin</p>
                            </div>
                        </div>
                        <SignOutButton>
                            <button className="p-2 text-gray-400 hover:text-gray-600 transition-colors">
                                <ArrowRightOnRectangleIcon className="h-5 w-5" />
                            </button>
                        </SignOutButton>
                    </div>
                </div>
            </div>

            {/* Main content */}
            <div className="lg:pl-64">
                {/* Top header */}
                <header className="bg-white shadow-sm border-b border-gray-200">
                    <div className="px-4 sm:px-6 lg:px-8">
                        <div className="flex items-center justify-between h-16">
                            <div className="flex items-center">
                                <h1 className="text-lg font-semibold text-gray-900">
                                    {navigation.find(item => isActive(item.href))?.name || "Dashboard"}
                                </h1>
                            </div>
                            <div className="flex items-center space-x-4">
                                <span className="text-sm text-gray-500">
                                    {new Date().toLocaleDateString()}
                                </span>
                            </div>
                        </div>
                    </div>
                </header>

                {/* Page content */}
                <main className="py-6">
                    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                        {children}
                    </div>
                </main>
            </div>
        </div>
    );
}