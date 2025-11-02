import { clerkMiddleware, createRouteMatcher } from "@clerk/nextjs/server";

const isPublicRoute = createRouteMatcher(["/", "/sign-in(.*)", "/sign-up(.*)"]);

export default clerkMiddleware((auth, req) => {
    if (!isPublicRoute(req)) {
        // For API routes, return 401 JSON instead of redirect
        if (req.nextUrl.pathname.startsWith('/api/')) {
            const { userId } = auth();
            if (!userId) {
                return new Response('Unauthorized', { status: 401 });
            }
        } else {
            auth().protect();
        }
    } else {
        auth().protect();
    }
});

export const config = {
    matcher: ["/((?!.+\\.[\\w]+$|_next).*)", "/", "/(api|trpc)(.*)"],
};