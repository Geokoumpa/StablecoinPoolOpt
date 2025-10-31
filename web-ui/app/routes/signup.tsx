import { SignUp, useUser } from "@clerk/clerk-react";
import { Link, useNavigate } from "react-router";
import { useEffect } from "react";

export default function SignUpPage() {
    const { isSignedIn, user } = useUser();
    const navigate = useNavigate();

    // Redirect to dashboard if already signed in
    useEffect(() => {
        if (isSignedIn) {
            navigate("/");
        }
    }, [isSignedIn, navigate]);

    return (
        <div className="min-h-screen flex items-center justify-center bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
            <div className="max-w-md w-full space-y-8">
                <div>
                    <div className="mx-auto h-12 w-12 flex items-center justify-center rounded-full bg-blue-100">
                        <svg className="h-8 w-8 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                        </svg>
                    </div>
                    <h2 className="mt-6 text-center text-3xl font-extrabold text-gray-900">
                        Create your account
                    </h2>
                    <p className="mt-2 text-center text-sm text-gray-600">
                        Sign up to access the Stablecoin Pool Optimization dashboard
                    </p>
                </div>

                <div className="mt-8">
                    <div className="bg-white py-8 px-4 shadow sm:rounded-lg sm:px-10">
                        <SignUp
                            path="/signup"
                            routing="path"
                            signInUrl="/login"
                            redirectUrl="/"
                            appearance={{
                                elements: {
                                    rootBox: "mx-auto",
                                    card: "shadow-none border-0"
                                }
                            }}
                        />
                    </div>
                </div>

                <div className="text-center">
                    <p className="text-sm text-gray-600">
                        Already have an account?{" "}
                        <Link to="/login" className="font-medium text-blue-600 hover:text-blue-500">
                            Sign in
                        </Link>
                    </p>
                </div>
            </div>
        </div>
    );
}