'use client';

import { SignUp } from "@clerk/nextjs";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import Link from "next/link";

export default function SignUpPage() {
    return (
        <div className="min-h-screen flex items-center justify-center bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
            <Card className="w-full max-w-md">
                <CardHeader>
                    <CardTitle>Create Account</CardTitle>
                    <CardDescription>
                        Sign up to access the stablecoin pool optimization dashboard
                    </CardDescription>
                </CardHeader>
                <CardContent>
                    <SignUp
                        redirectUrl="/dashboard"
                        signInUrl="/login"
                        appearance={{
                            elements: {
                                formButton: "w-full",
                            },
                        }}
                    />
                </CardContent>
            </Card>
        </div>
    );
}