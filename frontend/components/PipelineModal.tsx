import React, { useState } from "react";

interface PipelineModalProps {
    isOpen: boolean;
    onClose: () => void;
    onRun: (options: any) => void;
    videoName: string;
}

export function PipelineModal({ isOpen, onClose, onRun, videoName }: PipelineModalProps) {
    const [fps, setFps] = useState(5);
    const [visualize, setVisualize] = useState(true);
    const [smoothing, setSmoothing] = useState(true);
    const [overwrite, setOverwrite] = useState(false);

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
            <div className="w-full max-w-md rounded-xl bg-white p-6 shadow-2xl dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800">
                <h3 className="text-lg font-bold mb-1">Run Analysis Pipeline</h3>
                <p className="text-sm text-neutral-500 mb-6">Configuration for: <span className="font-mono text-neutral-700 dark:text-neutral-300">{videoName}</span></p>

                <div className="space-y-4">

                    <div>
                        <label className="block text-sm font-medium mb-1">Analysis FPS</label>
                        <input
                            type="number"
                            value={fps}
                            onChange={(e) => setFps(Number(e.target.value))}
                            className="w-full rounded-md border border-neutral-300 px-3 py-2 text-sm dark:border-neutral-700 dark:bg-black"
                            min={1} max={30}
                        />
                        <p className="text-xs text-neutral-400 mt-1">Frames per second to extract and analyze.</p>
                    </div>

                    <div className="flex items-center justify-between">
                        <label className="text-sm font-medium">Generate Visualization Video</label>
                        <input
                            type="checkbox"
                            checked={visualize}
                            onChange={(e) => setVisualize(e.target.checked)}
                            className="h-4 w-4 rounded border-neutral-300 text-black focus:ring-black"
                        />
                    </div>

                    <div className="flex items-center justify-between">
                        <label className="text-sm font-medium">Enable Smoothing</label>
                        <input
                            type="checkbox"
                            checked={smoothing}
                            onChange={(e) => setSmoothing(e.target.checked)}
                            className="h-4 w-4 rounded border-neutral-300 text-black focus:ring-black"
                        />
                    </div>

                    <div className="flex items-center justify-between">
                        <label className="text-sm font-medium text-red-600 dark:text-red-400">Overwrite Existing Results</label>
                        <input
                            type="checkbox"
                            checked={overwrite}
                            onChange={(e) => setOverwrite(e.target.checked)}
                            className="h-4 w-4 rounded border-neutral-300 text-red-600 focus:ring-red-600"
                        />
                    </div>

                </div>

                <div className="mt-8 flex justify-end gap-3">
                    <button
                        onClick={onClose}
                        className="rounded-md border border-neutral-200 px-4 py-2 text-sm font-medium text-neutral-600 hover:bg-neutral-50 dark:border-neutral-800 dark:text-neutral-400 dark:hover:bg-neutral-800"
                    >
                        Cancel
                    </button>
                    <button
                        onClick={() => onRun({ fps, visualize, smoothing, overwrite })}
                        className="rounded-md bg-black px-4 py-2 text-sm font-medium text-white hover:bg-neutral-800 dark:bg-white dark:text-black dark:hover:bg-neutral-200"
                    >
                        Run Pipeline
                    </button>
                </div>
            </div>
        </div>
    );
}
