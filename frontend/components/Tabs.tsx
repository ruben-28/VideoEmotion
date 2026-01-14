import React from "react";

interface Tab {
    id: string;
    label: string;
    count?: number;
}

interface TabsProps {
    tabs: Tab[];
    activeTab: string;
    onChange: (id: string) => void;
}

export function Tabs({ tabs, activeTab, onChange }: TabsProps) {
    return (
        <div className="border-b border-neutral-200 dark:border-neutral-800">
            <nav className="-mb-px flex space-x-8" aria-label="Tabs">
                {tabs.map((tab) => {
                    const isActive = activeTab === tab.id;
                    return (
                        <button
                            key={tab.id}
                            onClick={() => onChange(tab.id)}
                            className={`
                whitespace-nowrap border-b-2 py-4 px-1 text-sm font-medium transition-colors
                ${isActive
                                    ? "border-black text-black dark:border-white dark:text-white"
                                    : "border-transparent text-neutral-500 hover:border-neutral-300 hover:text-neutral-700 dark:text-neutral-400 dark:hover:border-neutral-700 dark:hover:text-neutral-300"
                                }
              `}
                            aria-current={isActive ? "page" : undefined}
                        >
                            {tab.label}
                            {tab.count !== undefined && (
                                <span
                                    className={`ml-3 rounded-full py-0.5 px-2.5 text-xs font-medium md:inline-block
                    ${isActive
                                            ? "bg-black text-white dark:bg-white dark:text-black"
                                            : "bg-neutral-100 text-neutral-900 dark:bg-neutral-800 dark:text-neutral-100"
                                        }
                  `}
                                >
                                    {tab.count}
                                </span>
                            )}
                        </button>
                    );
                })}
            </nav>
        </div>
    );
}
