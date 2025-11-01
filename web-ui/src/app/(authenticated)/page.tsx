export default function DashboardPage() {
    return (
        <div className="space-y-6">
            <div className="bg-white shadow rounded-lg p-6">
                <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
                <p className="mt-2 text-gray-600">
                    Welcome to the Stablecoin Pool Optimization admin interface
                </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <div className="bg-white shadow rounded-lg p-6">
                    <h3 className="text-lg font-medium text-gray-900 mb-2">Total Pools</h3>
                    <p className="text-3xl font-bold text-blue-600">0</p>
                    <p className="text-sm text-gray-500">Active pools</p>
                </div>

                <div className="bg-white shadow rounded-lg p-6">
                    <h3 className="text-lg font-medium text-gray-900 mb-2">Optimization Runs</h3>
                    <p className="text-3xl font-bold text-green-600">0</p>
                    <p className="text-sm text-gray-500">Total runs</p>
                </div>

                <div className="bg-white shadow rounded-lg p-6">
                    <h3 className="text-lg font-medium text-gray-900 mb-2">Total Value Locked</h3>
                    <p className="text-3xl font-bold text-purple-600">$0</p>
                    <p className="text-sm text-gray-500">Across all pools</p>
                </div>

                <div className="bg-white shadow rounded-lg p-6">
                    <h3 className="text-lg font-medium text-gray-900 mb-2">Average APY</h3>
                    <p className="text-3xl font-bold text-orange-600">0%</p>
                    <p className="text-sm text-gray-500">Weighted average</p>
                </div>
            </div>

            <div className="bg-white shadow rounded-lg p-6">
                <h2 className="text-lg font-medium text-gray-900 mb-4">Recent Activity</h2>
                <div className="text-center py-8 text-gray-500">
                    <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m0 0l-3-3m3 3V2m-6 4l-3 3m0 0l-3-3m3 3V2" />
                    </svg>
                    <p className="mt-2">No recent activity</p>
                </div>
            </div>
        </div>
    );
}