import React, { useEffect, useState } from "react";

interface PipelineProgress {
    current_step: string;
    current_step_index: number;
    total_steps: number;
    percent: number;
}

interface PipelineJob {
    job_id: string;
    video_name: string;
    status: "pending" | "running" | "done" | "error" | "cancelled";
    created_at: string;
    progress: PipelineProgress | null;
    logs: string[];
    error: string | null;
}

export function ActivePipelineJobs() {
    const [jobs, setJobs] = useState<PipelineJob[]>([]);

    const fetchJobs = async () => {
        try {
            const res = await fetch("http://localhost:8000/api/pipeline/jobs?limit=10");
            if (res.ok) {
                const data = await res.json();
                setJobs(data.jobs);
            }
        } catch (e) {
            // Quiet fail for polling
        }
    };

    useEffect(() => {
        fetchJobs();
        const interval = setInterval(fetchJobs, 1000);
        return () => clearInterval(interval);
    }, []);

    const activeJobs = jobs.filter((j) => ["running", "pending"].includes(j.status));

    if (activeJobs.length === 0) return null;

    return (
        <div className="grid gap-4 mb-8">
            {activeJobs.map((job) => (
                <div
                    key={job.job_id}
                    className="rounded-xl border border-neutral-200 bg-white p-4 shadow-sm dark:border-neutral-800 dark:bg-neutral-900"
                >
                    <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-3">
                            <div
                                className={`flex h-8 w-8 items-center justify-center rounded-full ${job.status === "running"
                                    ? "bg-blue-50 text-blue-600 dark:bg-blue-900/20 dark:text-blue-400"
                                    : "bg-yellow-50 text-yellow-600 dark:bg-yellow-900/20 dark:text-yellow-400"
                                    }`}
                            >
                                {job.status === "running" ? (
                                    <svg
                                        className="w-4 h-4 animate-spin"
                                        fill="none"
                                        viewBox="0 0 24 24"
                                    >
                                        <circle
                                            className="opacity-25"
                                            cx="12"
                                            cy="12"
                                            r="10"
                                            stroke="currentColor"
                                            strokeWidth="4"
                                        />
                                        <path
                                            className="opacity-75"
                                            fill="currentColor"
                                            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                                        />
                                    </svg>
                                ) : (
                                    <svg
                                        className="w-4 h-4"
                                        fill="none"
                                        viewBox="0 0 24 24"
                                        stroke="currentColor"
                                    >
                                        <path
                                            strokeLinecap="round"
                                            strokeLinejoin="round"
                                            strokeWidth={2}
                                            d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
                                        />
                                    </svg>
                                )}
                            </div>
                            <div>
                                <h4 className="font-medium text-sm">
                                    Processing: {job.video_name}
                                </h4>
                                <p className="text-xs text-neutral-500">
                                    {job.status === "running" ? "Analysis in progress..." : "Queued"}
                                </p>
                            </div>
                        </div>
                        <div className="text-right">
                            <span className="block text-2xl font-bold tracking-tight">
                                {Math.round(job.progress?.percent || 0)}%
                            </span>
                        </div>
                    </div>

                    <div className="space-y-2">
                        <div className="h-2 w-full rounded-full bg-neutral-100 dark:bg-neutral-800 overflow-hidden">
                            <div
                                className="h-full bg-black dark:bg-white transition-all duration-500 ease-out"
                                style={{ width: `${job.progress?.percent || 0}%` }}
                            />
                        </div>
                    </div>


                </div>
            ))}
        </div>
    );
}
