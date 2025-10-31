import { useParams } from "react-router";

export default function PoolDetails() {
    const { poolId } = useParams();

    return (
        <div className="p-4">
            <h1 className="text-2xl font-bold">Pool Details</h1>
            <p>Viewing pool: {poolId}</p>
        </div>
    );
}