"use client";

import { useEffect, useState, useMemo } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Tabs } from "@/components/Tabs";
import { VideoList } from "@/components/VideoList";
import { PipelineModal } from "@/components/PipelineModal";

// --- Types ---
interface Video {
  id: string;
  name: string;
  status: "processed" | "partial" | "unprocessed";
  mode: "offline" | "realtime";
  created_at: string;
  processed_at?: string;
  file_size_mb?: number;
}

interface Stats {
  total_videos: number;
  processed: number;
  unprocessed: number;
  total_size_mb: number;
  trash_stats: {
    count: number;
    size_mb: number;
  };
}

// --- Icons ---
const Icons = {
  Refresh: () => <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" /></svg>,
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
          Video Analysis
        </span>
      </div>
    </div>
  );
}

// --- Main Page ---

export default function Home() {
  const router = useRouter();

  // Data State
  const [videos, setVideos] = useState<Video[]>([]);
  const [trashItems, setTrashItems] = useState<Video[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // UI State
  const [activeTab, setActiveTab] = useState("all");
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [pipelineModalOpen, setPipelineModalOpen] = useState(false);
  const [targetVideo, setTargetVideo] = useState<Video | null>(null); // Video to run pipeline on (single)

  const API_BASE = "http://localhost:8000";

  // --- Fetching ---

  const loadData = async (withScan = false) => {
    try {
      setLoading(true);
      setError(null);

      if (withScan) {
        // Trigger a filesystem scan first
        try {
          await fetch(`${API_BASE}/api/videos/scan`, { method: "POST" });
          // Small delay to allow scan to process
          await new Promise(r => setTimeout(r, 500));
        } catch (e) {
          console.error("Scan failed", e);
        }
      }

      // 1. Fetch Videos
      const vidRes = await fetch(`${API_BASE}/api/videos?per_page=100&sort_by=created_at&sort_order=desc`);
      if (!vidRes.ok) throw new Error("Backend offline");
      const vidData = await vidRes.json();
      setVideos(vidData.videos || []);

      // 2. Fetch Trash
      const trashRes = await fetch(`${API_BASE}/api/trash`);
      if (trashRes.ok) {
        const trashData = await trashRes.json();
        // Map trash items to Video interface structure for compatibility
        const mappedTrash = trashData.trash_items.map((t: any) => ({
          id: t.trash_id,
          name: t.video_name,
          status: t.original_status, // Use preserved status
          mode: t.mode,
          created_at: t.deleted_at, // Use deleted_at for display
          file_size_mb: t.size_mb
        }));
        setTrashItems(mappedTrash);
      }

      // 3. Fetch Stats
      const statRes = await fetch(`${API_BASE}/api/stats`);
      if (statRes.ok) {
        const statData = await statRes.json();
        setStats(statData);
      }

    } catch (err) {
      console.error(err);
      setError("Could not connect to VideoEmotion API. Is it running on port 8000?");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  // --- Tabs Logic ---

  const filteredVideos = useMemo(() => {
    if (activeTab === "trash") return trashItems;
    if (activeTab === "all") return videos;
    if (activeTab === "processed") return videos.filter(v => v.status === "processed");
    if (activeTab === "unprocessed") return videos.filter(v => v.status === "unprocessed");
    if (activeTab === "partial") return videos.filter(v => v.status === "partial");
    return videos;
  }, [activeTab, videos, trashItems]);

  const tabs = [
    { id: "all", label: "All Videos", count: videos.length },
    { id: "processed", label: "Processed", count: videos.filter(v => v.status === "processed").length },
    { id: "partial", label: "Partial", count: videos.filter(v => v.status === "partial").length },
    { id: "unprocessed", label: "Unprocessed", count: videos.filter(v => v.status === "unprocessed").length },
    { id: "trash", label: "Trash", count: trashItems.length },
  ];

  // --- Selection Logic ---

  const handleToggleSelect = (id: string) => {
    const newSet = new Set(selectedIds);
    if (newSet.has(id)) {
      newSet.delete(id);
    } else {
      newSet.add(id);
    }
    setSelectedIds(newSet);
  };

  const handleToggleSelectAll = () => {
    if (selectedIds.size === filteredVideos.length) {
      setSelectedIds(new Set());
    } else {
      setSelectedIds(new Set(filteredVideos.map(v => v.id)));
    }
  };

  // --- Actions ---

  const handleDelete = async (video: Video) => {
    const isTrash = activeTab === "trash";
    const confirmMsg = isTrash
      ? `Are you sure you want to permanently delete "${video.name}"? This cannot be undone.`
      : `Are you sure you want to move "${video.name}" to trash?`;

    if (!confirm(confirmMsg)) return;

    try {
      const endpoint = isTrash ? `${API_BASE}/api/trash/${video.id}` : `${API_BASE}/api/videos/${video.id}`;
      const res = await fetch(endpoint, { method: "DELETE" });

      if (!res.ok) throw new Error("Delete failed");
      loadData(); // Refresh all lists
    } catch (err) {
      alert("Failed to delete video");
    }
  };

  const handleBatchDelete = async () => {
    const isTrash = activeTab === "trash";
    const confirmMsg = isTrash
      ? `Permanently delete ${selectedIds.size} videos? This cannot be undone.`
      : `Move ${selectedIds.size} videos to trash?`;

    if (!confirm(confirmMsg)) return;

    for (const id of Array.from(selectedIds)) {
      try {
        const endpoint = isTrash ? `${API_BASE}/api/trash/${id}` : `${API_BASE}/api/videos/${id}`;
        await fetch(endpoint, { method: "DELETE" });
      } catch (e) {
        console.error("Batch delete failed for", id);
      }
    }
    setSelectedIds(new Set());
    loadData();
  };

  const handleRestore = async (video: Video) => {
    try {
      const res = await fetch(`${API_BASE}/api/trash/${video.id}/restore`, { method: "POST" });
      if (!res.ok) throw new Error("Restore failed");
      loadData();
    } catch (err) {
      alert("Failed to restore video");
    }
  };

  const handleBatchRestore = async () => {
    for (const id of Array.from(selectedIds)) {
      await fetch(`${API_BASE}/api/trash/${id}/restore`, { method: "POST" });
    }
    setSelectedIds(new Set());
    loadData();
  };

  const handleProcessEmpty = async () => {
    // Trigger modal for single video
    // But user wants "run pipeline on ANY video"
    // If none selected, maybe show alert or just nothing
  }

  const openProcessModal = (video: Video) => {
    setTargetVideo(video);
    setPipelineModalOpen(true);
  }

  const runPipeline = async (options: any) => {
    if (!targetVideo && selectedIds.size === 0) return;

    const videosToRun = targetVideo ? [targetVideo] : filteredVideos.filter(v => selectedIds.has(v.id));

    setPipelineModalOpen(false); // Close immediately
    setTargetVideo(null);
    setSelectedIds(new Set()); // Clear selection

    alert(`Starting pipeline for ${videosToRun.length} videos...`);

    for (const vid of videosToRun) {
      try {
        await fetch(`${API_BASE}/api/pipeline/run`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ video_name: vid.name, options: options })
        });
      } catch (e) {
        console.error(`Failed to start pipeline for ${vid.name}`, e);
      }
    }
    // Reload mainly to update status badges or show queued
    loadData();
  }

  // Batch Process Button Handler
  const handleBatchProcessClick = () => {
    if (selectedIds.size === 0) return;
    // Use the first video as "representative" for the modal name display, or show "X Videos"
    const dummyVideo = { name: `${selectedIds.size} Selected Videos` } as Video;
    setTargetVideo(dummyVideo);
    setPipelineModalOpen(true);
  }


  return (
    <div className="min-h-screen bg-zinc-50 dark:bg-black font-sans text-neutral-900 dark:text-neutral-100">

      {/* Navbar */}
      <header className="sticky top-0 z-50 w-full border-b border-neutral-200 bg-black py-4 dark:border-neutral-800">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6">
          <Logo />
          <div className="flex items-center gap-4">
            <Link
              href="/analytics"
              className="text-sm font-medium text-neutral-400 hover:text-white transition-colors"
            >
              Analytics
            </Link>
            <button
              onClick={() => loadData(true)}
              className="rounded-full bg-white/10 p-2 text-white transition-colors hover:bg-white/20"
              title="Sync & Refresh Data"
            >
              <Icons.Refresh />
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-6 py-8">

        {/* Stats Grid */}
        {stats && (
          <div className="mb-12 grid gap-6 sm:grid-cols-2 lg:grid-cols-5">
            {[
              { label: "Total Videos", value: stats.total_videos },
              { label: "Processed", value: stats.processed },
              { label: "Unprocessed", value: stats.unprocessed },
              { label: "Trash", value: stats.trash_stats.count },
              { label: "Storage Used", value: `${(stats.total_size_mb / 1024).toFixed(1)} GB` },
            ].map((stat, i) => (
              <div key={i} className="rounded-xl border border-neutral-200 bg-white p-6 shadow-sm dark:border-neutral-800 dark:bg-neutral-900/50">
                <p className="text-sm font-medium text-neutral-500 dark:text-neutral-400">{stat.label}</p>
                <p className="mt-2 text-3xl font-bold tracking-tight">{stat.value}</p>
              </div>
            ))}
          </div>
        )}

        {/* Controls */}
        <div className="flex flex-col gap-6 sm:flex-row sm:items-center sm:justify-between mb-8">
          <h2 className="text-2xl font-bold tracking-tight">Videos</h2>

          {/* Batch Actions */}
          {selectedIds.size > 0 && (
            <div className="flex items-center gap-2 bg-neutral-100 px-4 py-2 rounded-lg dark:bg-neutral-900">
              <span className="text-sm font-medium">{selectedIds.size} selected</span>
              <div className="h-4 w-px bg-neutral-300 dark:bg-neutral-700 mx-2" />

              {activeTab === "trash" ? (
                <>
                  <button
                    onClick={handleBatchRestore}
                    className="text-sm text-neutral-600 hover:text-black dark:text-neutral-400 dark:hover:text-white"
                  >
                    Restore
                  </button>
                  <button
                    onClick={handleBatchDelete}
                    className="text-sm text-red-600 hover:text-red-700"
                  >
                    Delete Forever
                  </button>
                </>
              ) : (
                <>
                  <button
                    onClick={handleBatchProcessClick}
                    className="text-sm text-neutral-600 hover:text-black dark:text-neutral-400 dark:hover:text-white"
                  >
                    Process
                  </button>
                  <button
                    onClick={handleBatchDelete}
                    className="text-sm text-red-600 hover:text-red-700"
                  >
                    Delete
                  </button>
                </>
              )}
            </div>
          )}
        </div>

        {/* Tabs */}
        <Tabs tabs={tabs} activeTab={activeTab} onChange={(id) => { setActiveTab(id); setSelectedIds(new Set()); }} />

        {/* List */}
        <div className="mt-6">
          <VideoList
            videos={filteredVideos}
            selectedIds={selectedIds}
            onToggleSelect={handleToggleSelect}
            onToggleSelectAll={handleToggleSelectAll}
            onProcess={openProcessModal}
            onDelete={handleDelete}
            onRestore={activeTab === "trash" ? handleRestore : undefined}
            showRestore={activeTab === "trash"}
          />
        </div>

      </main>

      {/* Modal */}
      <PipelineModal
        isOpen={pipelineModalOpen}
        onClose={() => { setPipelineModalOpen(false); setTargetVideo(null); }}
        onRun={runPipeline}
        videoName={targetVideo?.name || "Multiple Videos"}
      />

    </div>
  );
}
