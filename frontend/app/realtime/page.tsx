"use client";

import { useState, useEffect, useCallback } from "react";
import Link from "next/link";

// --- Types ---

interface RealtimeConfig {
    camera_id: number;
    display_width: number;
    min_det_score: number;
    save_json: boolean;
    save_video: boolean;
    visualize: boolean;
}

interface RealtimeStatus {
    session_id: string;
    status: "idle" | "running" | "stopping" | "error";
    start_time: string;
    config: RealtimeConfig;
    output_dir?: string;
    error?: string;
}

// --- Icons ---

const Icons = {
    Refresh: () => <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" /></svg>,
    Play: () => <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>,
    Stop: () => <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z" /></svg>,
    Camera: () => <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" /></svg>,
};

function Logo() {
    return (
        <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-full border-2 border-white text-white font-bold tracking-tighter">
                VE
            </div>
            <div className="flex flex-col">
                <span className="text-xl font-bold tracking-tight text-white leading-none">
                    VideoEmotion
                </span>
                <span className="text-[10px] font-medium tracking-widest text-neutral-400 uppercase leading-none mt-1">
                    Realtime Analysis
                </span>
            </div>
        </div>
    );
}

export default function RealtimePage() {
    const [status, setStatus] = useState<RealtimeStatus | null>(null);
    const [logs, setLogs] = useState<string[]>([]);
    const [config, setConfig] = useState<RealtimeConfig>({
        camera_id: 0,
        display_width: 800,
        min_det_score: 0.65,
        save_json: true,
        save_video: true,
        visualize: true,
    });
    const [isLoading, setIsLoading] = useState(false);

    // Use local API Base for this standalone page (assuming same backend)
    const API_BASE = "http://localhost:8000";

    const fetchStatus = useCallback(async () => {
        try {
            const res = await fetch(`${API_BASE}/api/realtime/status`);
            const data = await res.json();
            setStatus(data);
        } catch (error) {
            console.error("Failed to fetch status:", error);
        }
    }, []);

    const fetchLogs = useCallback(async () => {
        try {
            const res = await fetch(`${API_BASE}/api/realtime/logs?limit=100`);
            const data = await res.json();
            setLogs(data.logs || []);
        } catch (error) {
            console.error("Failed to fetch logs:", error);
        }
    }, []);

    useEffect(() => {
        fetchStatus();
        const statusInterval = setInterval(fetchStatus, 2000);
        const logsInterval = setInterval(fetchLogs, 1000);

        return () => {
            clearInterval(statusInterval);
            clearInterval(logsInterval);
        };
    }, [fetchStatus, fetchLogs]);

    const handleStart = async () => {
        setIsLoading(true);
        try {
            const res = await fetch(`${API_BASE}/api/realtime/start`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(config),
            });

            if (!res.ok) {
                const error = await res.json();
                alert(`Failed to start: ${error.detail}`);
            } else {
                await fetchStatus();
            }
        } catch (error) {
            alert(`Error: ${error}`);
        } finally {
            setIsLoading(false);
        }
    };

    const handleStop = async () => {
        setIsLoading(true);
        try {
            await fetch(`${API_BASE}/api/realtime/stop`, {
                method: "POST",
            });
            await fetchStatus();
        } catch (error) {
            alert(`Error: ${error}`);
        } finally {
            setIsLoading(false);
        }
    };

    const isRunning = status?.status === "running";

    return (
        <div className="min-h-screen bg-zinc-50 dark:bg-black font-sans text-neutral-900 dark:text-neutral-100">

            {/* Navbar - Matches Main UI */}
            <header className="sticky top-0 z-50 w-full border-b border-neutral-200 bg-black py-4 dark:border-neutral-800">
                <div className="mx-auto flex max-w-7xl items-center justify-between px-6">
                    <Logo />
                    <div className="flex items-center gap-4">
                        <Link
                            href="/"
                            className="text-sm font-medium text-neutral-400 hover:text-white transition-colors"
                        >
                            Back to Dashboard
                        </Link>
                    </div>
                </div>
            </header>

            <main className="mx-auto max-w-7xl px-6 py-8">

                {/* Header Section */}
                <div className="flex flex-col gap-6 sm:flex-row sm:items-center sm:justify-between mb-8">
                    <div>
                        <h2 className="text-2xl font-bold tracking-tight">Session Control</h2>
                        <p className="text-neutral-500 dark:text-neutral-400">Manage your live analysis session.</p>
                    </div>

                    <div className="flex items-center gap-3">
                        <button
                            onClick={handleStart}
                            disabled={isRunning || isLoading}
                            className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium text-sm transition-all ${isRunning || isLoading
                                    ? "bg-neutral-100 text-neutral-400 dark:bg-neutral-800 dark:text-neutral-600 cursor-not-allowed"
                                    : "bg-black text-white hover:bg-neutral-800 dark:bg-white dark:text-black dark:hover:bg-neutral-200"
                                }`}
                        >
                            {isLoading && !isRunning ? (
                                <span>Starting...</span>
                            ) : (
                                <>
                                    <Icons.Play /> Start
                                </>
                            )}
                        </button>

                        <button
                            onClick={handleStop}
                            disabled={!isRunning || isLoading}
                            className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium text-sm transition-all ${!isRunning || isLoading
                                    ? "bg-neutral-100 text-neutral-400 dark:bg-neutral-800 dark:text-neutral-600 cursor-not-allowed"
                                    : "bg-red-600 text-white hover:bg-red-700"
                                }`}
                        >
                            {isLoading && isRunning ? (
                                <span>Stopping...</span>
                            ) : (
                                <>
                                    <Icons.Stop /> Stop
                                </>
                            )}
                        </button>
                    </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    {/* Configuration Card */}
                    <div className="lg:col-span-1 space-y-6">

                        {/* Status Widget */}
                        <div className="rounded-xl border border-neutral-200 bg-white p-6 shadow-sm dark:border-neutral-800 dark:bg-neutral-900/50">
                            <h3 className="text-base font-semibold mb-4">Status</h3>
                            <div className="space-y-4">
                                <div className="flex items-center justify-between">
                                    <span className="text-sm text-neutral-500">Current State</span>
                                    <div className="flex items-center gap-2">
                                        <span className={`relative flex h-2.5 w-2.5`}>
                                            <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${isRunning ? 'bg-green-400' : 'hidden'}`}></span>
                                            <span className={`relative inline-flex rounded-full h-2.5 w-2.5 ${isRunning ? 'bg-green-500' : 'bg-neutral-400'}`}></span>
                                        </span>
                                        <span className="text-sm font-medium capitalize">{status?.status || "Idle"}</span>
                                    </div>
                                </div>
                                {status?.session_id && (
                                    <div className="flex items-center justify-between">
                                        <span className="text-sm text-neutral-500">Session ID</span>
                                        <span className="text-xs font-mono bg-neutral-100 dark:bg-neutral-800 px-2 py-1 rounded">{status.session_id}</span>
                                    </div>
                                )}
                                {status?.error && (
                                    <div className="p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-900/50 rounded-lg text-sm text-red-600 dark:text-red-400">
                                        {status.error}
                                    </div>
                                )}
                            </div>
                        </div>

                        {/* Config Widget */}
                        <div className="rounded-xl border border-neutral-200 bg-white p-6 shadow-sm dark:border-neutral-800 dark:bg-neutral-900/50">
                            <h3 className="text-base font-semibold mb-4">Configuration</h3>
                            <div className="space-y-4">
                                <div>
                                    <label className="block text-xs font-medium text-neutral-500 mb-1.5 uppercase">Camera ID</label>
                                    <div className="relative">
                                        <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-neutral-400">
                                            <Icons.Camera />
                                        </div>
                                        <input
                                            type="number"
                                            value={config.camera_id}
                                            onChange={(e) => setConfig({ ...config, camera_id: parseInt(e.target.value) })}
                                            disabled={isRunning}
                                            className="pl-9 block w-full rounded-lg border border-neutral-200 bg-neutral-50 p-2.5 text-sm focus:border-black focus:ring-black disabled:opacity-50 dark:border-neutral-800 dark:bg-neutral-900 dark:focus:border-white"
                                        />
                                    </div>
                                </div>

                                <div>
                                    <label className="block text-xs font-medium text-neutral-500 mb-1.5 uppercase">Display Width ({config.display_width}px)</label>
                                    <input
                                        type="range"
                                        min="400"
                                        max="1920"
                                        step="100"
                                        value={config.display_width}
                                        onChange={(e) => setConfig({ ...config, display_width: parseInt(e.target.value) })}
                                        disabled={isRunning}
                                        className="w-full h-2 bg-neutral-200 rounded-lg appearance-none cursor-pointer dark:bg-neutral-700 accent-black dark:accent-white"
                                    />
                                </div>

                                <div>
                                    <label className="block text-xs font-medium text-neutral-500 mb-1.5 uppercase">Min Confidence ({config.min_det_score})</label>
                                    <input
                                        type="range"
                                        min="0.1"
                                        max="0.95"
                                        step="0.05"
                                        value={config.min_det_score}
                                        onChange={(e) => setConfig({ ...config, min_det_score: parseFloat(e.target.value) })}
                                        disabled={isRunning}
                                        className="w-full h-2 bg-neutral-200 rounded-lg appearance-none cursor-pointer dark:bg-neutral-700 accent-black dark:accent-white"
                                    />
                                </div>

                                <div className="space-y-2 pt-2">
                                    {[
                                        { key: "save_json", label: "Save JSON" },
                                        { key: "save_video", label: "Save Video" },
                                        { key: "visualize", label: "Visualize" },
                                    ].map((item) => (
                                        <label key={item.key} className="flex items-center gap-3 cursor-pointer">
                                            <div className="relative flex items-center">
                                                <input
                                                    type="checkbox"
                                                    checked={config[item.key as keyof RealtimeConfig] as boolean}
                                                    onChange={(e) => setConfig({ ...config, [item.key]: e.target.checked })}
                                                    disabled={isRunning}
                                                    className="peer h-4 w-4 rounded border-neutral-300 text-black focus:ring-black dark:border-neutral-700 dark:bg-neutral-800 dark:focus:ring-white"
                                                />
                                            </div>
                                            <span className="text-sm font-medium">{item.label}</span>
                                        </label>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Logs / Output */}
                    <div className="lg:col-span-2 space-y-6">
                        <div className="h-full min-h-[500px] rounded-xl border border-neutral-200 bg-black p-0 shadow-sm overflow-hidden flex flex-col">
                            <div className="flex items-center justify-between border-b border-neutral-800 bg-neutral-900/50 px-4 py-3">
                                <h3 className="text-sm font-medium text-neutral-200">Terminal Output</h3>
                                <button onClick={fetchLogs} className="text-xs text-neutral-400 hover:text-white flex items-center gap-1">
                                    <Icons.Refresh /> Refresh
                                </button>
                            </div>
                            <div className="flex-1 overflow-auto p-4 font-mono text-xs text-neutral-300 space-y-1">
                                {logs.length === 0 ? (
                                    <div className="h-full flex items-center justify-center text-neutral-600 italic">
                                        Ready to start. Logs will appear here.
                                    </div>
                                ) : (
                                    logs.map((log, i) => (
                                        <div key={i} className="break-all hover:bg-white/5 px-2 py-0.5 rounded">
                                            <span className="text-neutral-500 mr-2">[{new Date().toLocaleTimeString()}]</span>
                                            {log}
                                        </div>
                                    ))
                                )}
                            </div>
                        </div>
                    </div>
                </div>

            </main>
        </div>
    );
}
