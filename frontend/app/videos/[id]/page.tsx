"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer
} from "recharts";

interface VideoDetails {
    id: string;
    name: string;
    status: string;
    mode: string;
    created_at: string;
    file_size_mb: number;
    stats?: {
        avg_emotion: Record<string, number>;
        dominant_emotion: string;
        timeline: Array<{ timestamp: number; emotions: Record<string, number> }>;
    };
    video_url?: string;
    thumbnail_url?: string;
}

const EMOTION_COLORS: Record<string, string> = {
    happiness: "#facc15", // yellow-400
    sadness: "#60a5fa", // blue-400
    anger: "#f87171", // red-400
    surprise: "#4ade80", // green-400
    fear: "#c084fc", // purple-400
    disgust: "#2dd4bf", // teal-400
    neutral: "#9ca3af", // gray-400
};

export default function VideoDetailPage() {
    const params = useParams();
    const router = useRouter();
    const id = params.id as string;

    const [video, setVideo] = useState<VideoDetails | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const API_BASE = "http://localhost:8000";

    useEffect(() => {
        const fetchVideo = async () => {
            try {
                setLoading(true);
                const res = await fetch(`${API_BASE}/api/videos/${id}`);
                if (!res.ok) throw new Error("Video not found");
                const data = await res.json();
                setVideo(data);
            } catch (err) {
                setError(err instanceof Error ? err.message : "Failed to load video");
            } finally {
                setLoading(false);
            }
        };

        if (id) fetchVideo();
    }, [id]);

    if (loading) {
        return (
            <div className="flex min-h-screen items-center justify-center bg-zinc-50 dark:bg-black text-neutral-500">
                <span className="animate-pulse">Loading video details...</span>
            </div>
        );
    }

    if (error || !video) {
        return (
            <div className="flex min-h-screen flex-col items-center justify-center gap-4 bg-zinc-50 dark:bg-black text-neutral-900 dark:text-neutral-100">
                <h1 className="text-2xl font-bold">Error</h1>
                <p className="text-neutral-500">{error || "Video not found"}</p>
                <Link
                    href="/"
                    className="rounded-md bg-black px-4 py-2 text-sm font-medium text-white hover:bg-neutral-800 dark:bg-white dark:text-black dark:hover:bg-neutral-200"
                >
                    Back to Dashboard
                </Link>
            </div>
        );
    }

    // Use backend-provided URL with fallback
    const videoUrl = video.video_url ? `${API_BASE}${video.video_url}` : `${API_BASE}/static/videos/${video.name}/${video.name}.mp4`;
    const posterUrl = video.thumbnail_url ? `${API_BASE}${video.thumbnail_url}` : `${API_BASE}/static/output/thumbnails/${video.name}.jpg`;
    const isAnnotated = videoUrl.includes("annotated");

    return (
        <div className="min-h-screen bg-zinc-50 dark:bg-black font-sans text-neutral-900 dark:text-neutral-100">

            {/* Header */}
            <header className="sticky top-0 z-50 w-full border-b border-neutral-200 bg-white/80 backdrop-blur-md dark:border-neutral-800 dark:bg-black/80">
                <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
                    <Link href="/" className="flex items-center gap-2 text-sm font-medium text-neutral-500 hover:text-black dark:hover:text-white transition-colors">
                        ← Back to Dashboard
                    </Link>
                    <div className="font-semibold">{video.name}</div>
                    <div className="w-24" /> {/* Spacer */}
                </div>
            </header>

            <main className="mx-auto max-w-7xl px-6 py-8">
                <div className="grid gap-8 lg:grid-cols-3">

                    {/* Main Content: Player & Timeline */}
                    <div className="lg:col-span-2 space-y-8">

                        {/* Video Player */}
                        <div className="overflow-hidden rounded-xl border border-neutral-200 bg-black shadow-lg dark:border-neutral-800 relative group">
                            {isAnnotated && (
                                <div className="absolute top-4 left-4 z-10 rounded-md bg-black/70 px-2 py-1 text-xs font-medium text-white backdrop-blur-sm pointer-events-none">
                                    Running Analysis Visualization
                                </div>
                            )}
                            <video
                                src={videoUrl}
                                controls
                                className="w-full aspect-video"
                                poster={posterUrl}

                            >
                                Your browser does not support the video tag.
                            </video>
                        </div>

                        {/* Analysis Graph */}
                        {video.status === "processed" && video.stats?.timeline && (
                            <div className="rounded-xl border border-neutral-200 bg-white p-6 shadow-sm dark:border-neutral-800 dark:bg-neutral-900">
                                <h3 className="text-lg font-bold mb-4">Emotion Timeline</h3>
                                <div className="h-80 w-full">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <LineChart
                                            data={video.stats.timeline.map(item => ({
                                                ...item,
                                                emotions: Object.fromEntries(
                                                    Object.entries(item.emotions).map(([k, v]) => [k.toLowerCase(), v])
                                                )
                                            }))}
                                            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                                        >
                                            <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.2} />
                                            <XAxis
                                                dataKey="timestamp"
                                                stroke="#9ca3af"
                                                tickFormatter={(s) => {
                                                    const min = Math.floor(s / 60);
                                                    const sec = s % 60;
                                                    return `${min}:${sec.toString().padStart(2, '0')}`;
                                                }}
                                            />
                                            <YAxis stroke="#9ca3af" />
                                            <Tooltip
                                                contentStyle={{ backgroundColor: "#171717", borderColor: "#262626", color: "#fff" }}
                                                itemStyle={{ color: "#fff" }}
                                                labelFormatter={(label) => `Time: ${label}s`}
                                            />
                                            <Legend />
                                            {Object.entries(EMOTION_COLORS).map(([emotion, color]) => (
                                                <Line
                                                    key={emotion}
                                                    type="monotone"
                                                    dataKey={`emotions.${emotion}`}
                                                    name={emotion}
                                                    stroke={color}
                                                    strokeWidth={2}
                                                    dot={false}
                                                    activeDot={{ r: 4 }}
                                                />
                                            ))}
                                        </LineChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Sidebar: Metadata & Stats */}
                    <div className="space-y-6">

                        <div className="rounded-xl border border-neutral-200 bg-white p-6 shadow-sm dark:border-neutral-800 dark:bg-neutral-900">
                            <h3 className="text-lg font-bold mb-4">Metadata</h3>
                            <div className="space-y-4 text-sm">
                                <div className="flex justify-between py-2 border-b border-neutral-100 dark:border-neutral-800">
                                    <span className="text-neutral-500">Status</span>
                                    <span className="font-medium capitalize">{video.status}</span>
                                </div>
                                <div className="flex justify-between py-2 border-b border-neutral-100 dark:border-neutral-800">
                                    <span className="text-neutral-500">Mode</span>
                                    <span className="font-medium capitalize">{video.mode}</span>
                                </div>
                                <div className="flex justify-between py-2 border-b border-neutral-100 dark:border-neutral-800">
                                    <span className="text-neutral-500">Size</span>
                                    <span className="font-medium">{video.file_size_mb?.toFixed(2)} MB</span>
                                </div>
                                <div className="flex justify-between py-2 border-b border-neutral-100 dark:border-neutral-800">
                                    <span className="text-neutral-500">Created</span>
                                    <span className="font-medium">{new Date(video.created_at).toLocaleDateString()}</span>
                                </div>
                            </div>
                        </div>

                        {video.stats && (
                            <div className="rounded-xl border border-neutral-200 bg-white p-6 shadow-sm dark:border-neutral-800 dark:bg-neutral-900">
                                <h3 className="text-lg font-bold mb-4">Emotions</h3>
                                <div className="space-y-3">
                                    {Object.entries(video.stats.avg_emotion || {})
                                        .map(([k, v]) => [k.toLowerCase(), v] as const)
                                        .sort(([, a], [, b]) => b - a)
                                        .slice(0, 5)
                                        .map(([emotion, score]) => (
                                            <div key={emotion}>
                                                <div className="flex justify-between text-xs mb-1">
                                                    <span className="capitalize font-medium">{emotion}</span>
                                                    <span className="text-neutral-500">{(score * 100).toFixed(1)}%</span>
                                                </div>
                                                <div className="h-2 w-full rounded-full bg-neutral-100 dark:bg-neutral-800 overflow-hidden">
                                                    <div
                                                        className="h-full bg-black dark:bg-white rounded-full transition-all duration-500"
                                                        style={{ width: `${score * 100}%` }}
                                                    />
                                                </div>
                                            </div>
                                        ))}
                                </div>
                            </div>
                        )}

                        <div className="flex flex-col gap-2">
                            <button
                                onClick={async () => {
                                    if (confirm("Delete this video?")) {
                                        try {
                                            const res = await fetch(`${API_BASE}/api/videos/${id}`, { method: "DELETE" });
                                            if (!res.ok) {
                                                const error = await res.json().catch(() => ({ message: "Unknown error" }));
                                                alert(`Failed to delete video: ${error.message || res.statusText}`);
                                                return;
                                            }
                                            router.push("/");
                                        } catch (err) {
                                            alert(`Failed to delete video: ${err instanceof Error ? err.message : "Network error"}`);
                                        }
                                    }
                                }}
                                className="w-full rounded-lg border border-red-200 bg-red-50 py-2.5 text-sm font-medium text-red-600 hover:bg-red-100 dark:border-red-900/30 dark:bg-red-950/10 dark:text-red-400 dark:hover:bg-red-950/20 transition-colors"
                            >
                                Delete Video
                            </button>
                        </div>

                    </div>
                </div>
            </main>
        </div>
    );
}
