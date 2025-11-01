'use client';

import { useUser } from "@clerk/nextjs";
import { useRouter } from "next/navigation";

interface ProtectedRouteProps {
    children: React.ReactNode;
}

export function ProtectedRoute({ children }: ProtectedRouteProps) {
    const { isSignedIn, isLoaded } = useUser();
    const router = useRouter();

    if (!isLoaded) {
        return (
            <div className="min-h-screen flex items-center justify-center">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
            </div>
        );
    }

    if (!isSignedIn) {
        router.push("/login");
        return null;
    }

    return <>{children}</>;
}

// Hook to check if user is authenticated
export function useAuth() {
    const { isSignedIn, user, isLoaded } = useUser();

    return {
        isAuthenticated: isSignedIn,
        user,
        isLoading: !isLoaded,
    };
}