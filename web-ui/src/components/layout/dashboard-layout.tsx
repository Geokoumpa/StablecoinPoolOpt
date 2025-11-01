'use client';

import { useAuth } from "@clerk/nextjs";
import { Sidebar } from "@/components/layout/sidebar";
import { Header } from "@/components/layout/header";

export function DashboardLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    const { userId } = useAuth();

    if (!userId) {
        return (
            <div className="min-h-screen flex items-center justify-center bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
                <div className="max-w-md w-full text-center">
                    <h1 className="text-2xl font-bold text-gray-900">Please sign in</h1>
                    <p className="mt-2 text-gray-600">You need to be authenticated to access this page.</p>
                </div>
            </div>
        );
    }

    return (
        <div className="flex h-screen bg-gray-100">
            <Sidebar />
            <div className="flex-1 flex-col">
                <Header />
                <main className="flex-1 p-6">
                    {children}
                </main>
            </div>
        </div>
    );
}