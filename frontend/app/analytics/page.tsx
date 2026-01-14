"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
    PieChart,
    Pie,
    Cell,
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer
} from "recharts";

// --- Types ---
interface Stats {
    total_videos: number;
    processed: number;
    partial: number;
    unprocessed: number;
    total_size_mb: number;
    trash_stats?: {
        total_items: number;
        total_size_mb: number;
    };
    emotion_distribution?: Record<string, number>;
}

interface Video {
    id: string;
    name: string;
    file_size_mb: number;
    stats?: {
        dominant_emotion: string;
        global_distribution: Record<string, number>;
    };
}

// --- Colors ---
const EMOTION_COLORS: Record<string, string> = {
    happiness: "#facc15", // yellow-400
    sadness: "#60a5fa", // blue-400
    anger: "#f87171", // red-400
    surprise: "#4ade80", // green-400
    fear: "#c084fc", // purple-400
    disgust: "#2dd4bf", // teal-400
    neutral: "#9ca3af", // gray-400
    Unknown: "#525252", // neutral-600
};

export default function AnalyticsPage() {
    const [stats, setStats] = useState<Stats | null>(null);
    const [videos, setVideos] = useState<Video[]>([]);
    const [loading, setLoading] = useState(true);

    const API_BASE = "http://localhost:8000";

    useEffect(() => {
        const loadData = async () => {
            try {
                setLoading(true);
                // Fetch Global Stats
                const statsRes = await fetch(`${API_BASE}/api/stats`);
                const statsData = await statsRes.json();
                setStats(statsData);

                // Fetch All Videos for comparison
                const videosRes = await fetch(`${API_BASE}/api/videos?per_page=100`);
                const videosData = await videosRes.json();
                // Filter client-side to be robust against backend filter issues
                const processed = (videosData.videos || []).filter((v: Video) => v.stats);
                setVideos(processed);

            } catch (err) {
                console.error("Failed to load analytics data", err);
            } finally {
                setLoading(false);
            }
        };

        loadData();
    }, []);

    // Prepare Pie Chart Data
    const pieData = stats?.emotion_distribution
        ? Object.entries(stats.emotion_distribution)
            .map(([name, value]) => ({ name: name.toLowerCase(), value: value * 100 }))
            .sort((a, b) => b.value - a.value)
        : [];

    // Prepare Bar Chart Data (Top Emotions per Video)
    const barData = videos.slice(0, 10).map(v => {
        const rawDist = v.stats?.global_distribution || {};
        const dist: Record<string, number> = {};
        // Normalize keys to lowercase for matching
        Object.entries(rawDist).forEach(([k, val]) => {
            dist[k.toLowerCase()] = val;
        });

        return {
            name: v.name.length > 15 ? v.name.substring(0, 15) + "..." : v.name,
            happiness: (dist.happiness || 0) * 100,
            sadness: (dist.sadness || 0) * 100,
            anger: (dist.anger || 0) * 100,
            neutral: (dist.neutral || 0) * 100,
            full_name: v.name
        };
    });

    if (loading) {
        return (
            <div className="flex min-h-screen items-center justify-center bg-zinc-50 dark:bg-black text-neutral-500">
                <span className="animate-pulse">Loading analytics...</span>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-zinc-50 dark:bg-black font-sans text-neutral-900 dark:text-neutral-100">
            {/* Header */}
            <header className="sticky top-0 z-50 w-full border-b border-neutral-200 bg-white/80 backdrop-blur-md dark:border-neutral-800 dark:bg-black/80">
                <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
                    <Link href="/" className="flex items-center gap-2 text-sm font-medium text-neutral-500 hover:text-black dark:hover:text-white transition-colors">
                        ← Back to Dashboard
                    </Link>
                    <div className="font-semibold">Global Analytics</div>
                    <div className="w-24" />
                </div>
            </header>

            <main className="mx-auto max-w-7xl px-6 py-8">

                <h2 className="text-2xl font-bold tracking-tight mb-8">VideoEmotion Insights</h2>

                <div className="grid gap-8 lg:grid-cols-2 mb-12">

                    {/* Global Distribution Pie Chart */}
                    <div className="rounded-xl border border-neutral-200 bg-white p-6 shadow-sm dark:border-neutral-800 dark:bg-neutral-900">
                        <h3 className="text-lg font-bold mb-4">Overall Emotion Distribution</h3>
                        <div className="h-80 w-full">
                            <ResponsiveContainer width="100%" height="100%">
                                <PieChart>
                                    <Pie
                                        data={pieData}
                                        cx="50%"
                                        cy="50%"
                                        innerRadius={60}
                                        outerRadius={100}
                                        paddingAngle={5}
                                        dataKey="value"
                                    >
                                        {pieData.map((entry, index) => (
                                            <Cell key={`cell-${index}`} fill={EMOTION_COLORS[entry.name] || "#8884d8"} />
                                        ))}
                                    </Pie>
                                    <Tooltip
                                        formatter={(value: any) => `${Number(value).toFixed(1)}%`}
                                        contentStyle={{ backgroundColor: "#09090b", borderColor: "#27272a", borderRadius: "8px", boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)", color: "#fafafa" }}
                                        itemStyle={{ color: "#fafafa", fontWeight: 500 }}
                                    />
                                    <Legend />
                                </PieChart>
                            </ResponsiveContainer>
                        </div>
                    </div>


                    {/* Summary Stats Cards */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div className="rounded-xl border border-neutral-200 bg-white p-6 shadow-sm dark:border-neutral-800 dark:bg-neutral-900 flex flex-col justify-center">
                            <span className="text-sm text-neutral-500">Processed Videos</span>
                            <span className="text-4xl font-bold mt-2">{stats?.processed}</span>
                        </div>
                        <div className="rounded-xl border border-neutral-200 bg-white p-6 shadow-sm dark:border-neutral-800 dark:bg-neutral-900 flex flex-col justify-center">
                            <span className="text-sm text-neutral-500">Total Size</span>
                            <span className="text-4xl font-bold mt-2">{(stats?.total_size_mb || 0).toFixed(1)} MB</span>
                        </div>
                        <div className="rounded-xl border border-neutral-200 bg-white p-6 shadow-sm dark:border-neutral-800 dark:bg-neutral-900 flex flex-col justify-center">
                            <span className="text-sm text-neutral-500">Trash</span>
                            <span className="text-4xl font-bold mt-2">
                                {stats?.trash_stats?.total_items || 0}
                                <span className="text-lg font-normal text-neutral-400 ml-2">
                                    items
                                </span>
                            </span>
                            <div className="text-xs text-neutral-400 mt-1">
                                {(stats?.trash_stats?.total_size_mb || 0).toFixed(1)} MB recoverable
                            </div>
                        </div>
                        <div className="col-span-1 md:col-span-3 rounded-xl border border-neutral-200 bg-white p-6 shadow-sm dark:border-neutral-800 dark:bg-neutral-900">

                            <h4 className="text-sm font-medium text-neutral-500 mb-4">Dominant Emotions Breakdown</h4>
                            <div className="space-y-3">
                                {pieData.slice(0, 5).map((d) => (
                                    <div key={d.name} className="flex items-center gap-3">
                                        <div className="w-24 text-sm font-medium capitalize">{d.name}</div>
                                        <div className="flex-1 h-2 bg-neutral-100 rounded-full overflow-hidden dark:bg-neutral-800">
                                            <div
                                                className="h-full rounded-full"
                                                style={{ width: `${d.value}%`, backgroundColor: EMOTION_COLORS[d.name] || "#525252" }}
                                            />
                                        </div>
                                        <div className="w-12 text-sm text-right text-neutral-500">{d.value.toFixed(1)}%</div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>

                </div>

                {/* Comparison Bar Chart */}
                <div className="rounded-xl border border-neutral-200 bg-white p-6 shadow-sm dark:border-neutral-800 dark:bg-neutral-900">
                    <h3 className="text-lg font-bold mb-6">Compare Recent Videos</h3>
                    <div className="h-96 w-full">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart
                                data={barData}
                                layout="vertical"
                                margin={{ top: 5, right: 30, left: 40, bottom: 5 }}
                            >
                                <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#374151" opacity={0.2} />
                                <XAxis type="number" stroke="#9ca3af" unit="%" />
                                <YAxis dataKey="name" type="category" stroke="#9ca3af" width={100} />
                                <Tooltip
                                    cursor={{ fill: 'transparent' }}
                                    contentStyle={{ backgroundColor: "#09090b", borderColor: "#27272a", color: "#fafafa", borderRadius: "8px" }}
                                    formatter={(value: any) => `${Number(value).toFixed(1)}%`}
                                    labelFormatter={(label, payload) => payload[0]?.payload.full_name || label}
                                    itemSorter={(item) => (item.value as number)}
                                />
                                <Legend />
                                <Bar dataKey="happiness" stackId="a" fill={EMOTION_COLORS.happiness} />
                                <Bar dataKey="sadness" stackId="a" fill={EMOTION_COLORS.sadness} />
                                <Bar dataKey="anger" stackId="a" fill={EMOTION_COLORS.anger} />
                                <Bar dataKey="neutral" stackId="a" fill={EMOTION_COLORS.neutral} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>

            </main>
        </div>
    );
}
