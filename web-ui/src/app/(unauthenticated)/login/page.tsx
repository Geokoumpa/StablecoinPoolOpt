'use client';

import { SignIn } from "@clerk/nextjs";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import Link from "next/link";

export default function LoginPage() {
    return (
        <div className="min-h-screen flex items-center justify-center bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
            <Card className="w-full max-w-md">
                <CardHeader>
                    <CardTitle>Welcome Back</CardTitle>
                    <CardDescription>
                        Sign in to your account to access the stablecoin pool optimization dashboard
                    </CardDescription>
                </CardHeader>
                <CardContent>
                    <SignIn
                        redirectUrl="/"
                        signUpUrl="/signup"
                        appearance={{
                            elements: {
                                formButton: "w-full",
                            },
                        }}
                    />
                </CardContent>
            </Card>
            <p className="mt-6 text-center text-sm text-gray-600">
                Don't have an account?{" "}
                <Link
                    href="/signup"
                    className="font-medium text-blue-600 hover:text-blue-500"
                >
                    Sign up
                </Link>
            </p>
        </div>
    );
}