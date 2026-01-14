import React from "react";
import { useRouter } from "next/navigation";

interface Video {
    id: string;
    name: string;
    status: "processed" | "partial" | "unprocessed";
    mode: "offline" | "realtime";
    created_at: string;
    file_size_mb?: number;
}

interface VideoListProps {
    videos: Video[];
    selectedIds: Set<string>;
    onToggleSelect: (id: string) => void;
    onToggleSelectAll: () => void;
    onProcess: (video: Video) => void;
    onDelete: (video: Video) => void;
    onRestore?: (video: Video) => void; // Optional, for trash
    showRestore?: boolean;
}

// Icons (copied from page.tsx for consistency)
const Icons = {
    Play: () => <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>,
    Eye: () => <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" /></svg>,
    Trash: () => <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg>,
    Film: () => <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 4v16M17 4v16M3 8h4m10 0h4M3 12h18M3 16h4m10 0h4M4 20h16a1 1 0 001-1V5a1 1 0 00-1-1H4a1 1 0 00-1 1v14a1 1 0 001 1z" /></svg>,
    Refresh: () => <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" /></svg>,
};

function StatusBadge({ status }: { status: Video["status"] }) {
    let dotColor = "bg-zinc-400";
    let label = "Unprocessed";

    if (status === "processed") {
        dotColor = "bg-emerald-500";
        label = "Processed";
    } else if (status === "partial") {
        dotColor = "bg-amber-500";
        label = "Partial";
    }

    return (
        <span className="inline-flex items-center gap-2 rounded-full border border-neutral-200 bg-white px-2.5 py-0.5 text-xs font-medium text-neutral-700 dark:border-neutral-800 dark:bg-neutral-900 dark:text-neutral-300">
            <span className={`h-2 w-2 rounded-full ${dotColor} shadow-sm`} />
            {label}
        </span>
    );
}

export function VideoList({
    videos,
    selectedIds,
    onToggleSelect,
    onToggleSelectAll,
    onProcess,
    onDelete,
    onRestore,
    showRestore = false
}: VideoListProps) {
    const router = useRouter();
    const allSelected = videos.length > 0 && selectedIds.size === videos.length;

    if (videos.length === 0) {
        return (
            <div className="rounded-xl border border-neutral-200 bg-white p-12 text-center text-neutral-500 shadow-sm dark:border-neutral-800 dark:bg-neutral-900">
                No videos found in this category.
            </div>
        );
    }

    return (
        <div className="overflow-hidden rounded-xl border border-neutral-200 bg-white shadow-sm dark:border-neutral-800 dark:bg-neutral-900">
            <table className="w-full text-left text-sm">
                <thead className="border-b border-neutral-200 bg-neutral-50 dark:border-neutral-800 dark:bg-neutral-900/50">
                    <tr>
                        <th className="w-12 px-6 py-4">
                            <input
                                type="checkbox"
                                checked={allSelected}
                                onChange={onToggleSelectAll}
                                className="h-4 w-4 rounded border-neutral-300 text-black focus:ring-black dark:border-neutral-700 dark:bg-neutral-800"
                            />
                        </th>
                        <th className="px-6 py-4 font-semibold text-neutral-900 dark:text-neutral-100">Name</th>
                        <th className="px-6 py-4 font-semibold text-neutral-900 dark:text-neutral-100">Status</th>
                        <th className="px-6 py-4 font-semibold text-neutral-900 dark:text-neutral-100">Details</th>
                        <th className="px-6 py-4 text-right font-semibold text-neutral-900 dark:text-neutral-100">Actions</th>
                    </tr>
                </thead>
                <tbody className="divide-y divide-neutral-200 dark:divide-neutral-800">
                    {videos.map((video) => {
                        const isSelected = selectedIds.has(video.id);
                        return (
                            <tr
                                key={video.id}
                                className={`
                  group transition-colors
                  ${isSelected ? "bg-neutral-50 dark:bg-neutral-800/80" : "hover:bg-neutral-50 dark:hover:bg-neutral-800/50"}
                `}
                                onClick={() => onToggleSelect(video.id)} // Row click selects too
                            >
                                <td className="px-6 py-4" onClick={(e) => e.stopPropagation()}>
                                    <input
                                        type="checkbox"
                                        checked={isSelected}
                                        onChange={() => onToggleSelect(video.id)}
                                        className="h-4 w-4 rounded border-neutral-300 text-black focus:ring-black dark:border-neutral-700 dark:bg-neutral-800"
                                    />
                                </td>
                                <td className="px-6 py-4">
                                    <div className="flex items-center gap-3">
                                        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-neutral-100 text-neutral-500 dark:bg-neutral-800">
                                            <Icons.Film />
                                        </div>
                                        <div>
                                            <div className="font-medium text-neutral-900 dark:text-white">{video.name}</div>
                                            <div className="text-xs text-neutral-500 uppercase tracking-wide">{video.mode}</div>
                                        </div>
                                    </div>
                                </td>
                                <td className="px-6 py-4">
                                    <StatusBadge status={video.status} />
                                </td>
                                <td className="px-6 py-4 text-neutral-500">
                                    <div className="flex flex-col gap-1 text-xs">
                                        <span>Added: {new Date(video.created_at).toLocaleDateString()}</span>
                                        {video.file_size_mb && <span>Size: {video.file_size_mb.toFixed(1)} MB</span>}
                                    </div>
                                </td>
                                <td className="px-6 py-4" onClick={(e) => e.stopPropagation()}>
                                    <div className="flex justify-end gap-2 opacity-100 sm:opacity-0 sm:group-hover:opacity-100 transition-opacity">

                                        {showRestore && onRestore ? (
                                            <button
                                                onClick={() => onRestore(video)}
                                                className="rounded-md border border-neutral-200 bg-white px-3 py-1.5 text-xs font-medium text-neutral-700 hover:bg-neutral-50 dark:border-neutral-800 dark:bg-neutral-900 dark:text-neutral-300 dark:hover:bg-neutral-800"
                                            >
                                                Restore
                                            </button>
                                        ) : (
                                            <>
                                                <button
                                                    onClick={() => onProcess(video)}
                                                    className="rounded-md border border-neutral-200 p-2 text-neutral-600 hover:bg-neutral-50 hover:text-black dark:border-neutral-800 dark:text-neutral-400 dark:hover:bg-neutral-800 dark:hover:text-white"
                                                    title="Run Pipeline"
                                                >
                                                    <Icons.Play />
                                                </button>

                                                <button
                                                    onClick={() => router.push(`/videos/${video.id}`)}
                                                    className="rounded-md border border-neutral-200 p-2 text-neutral-600 hover:bg-neutral-50 hover:text-black dark:border-neutral-800 dark:text-neutral-400 dark:hover:bg-neutral-800 dark:hover:text-white"
                                                    title="View Details"
                                                >
                                                    <Icons.Eye />
                                                </button>
                                            </>
                                        )}

                                        <button
                                            onClick={() => onDelete(video)}
                                            className="rounded-md p-2 text-neutral-400 hover:bg-red-50 hover:text-red-600 transition-colors dark:hover:bg-red-950/20"
                                            title={showRestore ? "Delete Forever" : "Move to Trash"}
                                        >
                                            <Icons.Trash />
                                        </button>
                                    </div>
                                </td>
                            </tr>
                        );
                    })}
                </tbody>
            </table>
        </div>
    );
}
