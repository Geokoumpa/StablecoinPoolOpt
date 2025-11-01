'use client';

import { UserButton } from "@clerk/nextjs";

export function Header() {
    return (
        <header className="bg-white shadow-sm border-b border-gray-200">
            <div className="flex h-16 items-center justify-between px-6">
                <div className="flex-1">
                    <h1 className="text-xl font-semibold text-gray-900">
                        Stablecoin Pool Optimization
                    </h1>
                </div>

                <div className="flex items-center space-x-4">
                    <UserButton
                        afterSignOutUrl="/"
                        appearance={{
                            elements: {
                                avatarBox: "w-10 h-10",
                            },
                        }}
                    />
                </div>
            </div>
        </header>
    );
}