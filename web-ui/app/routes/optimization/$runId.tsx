import { useParams } from "react-router";

export default function OptimizationRun() {
    const { runId } = useParams();

    return (
        <div className="p-4">
            <h1 className="text-2xl font-bold">Optimization Run Details</h1>
            <p>Viewing run: {runId}</p>
        </div>
    );
}