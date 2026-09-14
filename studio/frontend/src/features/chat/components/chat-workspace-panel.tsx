// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useMemo, useState, type ReactElement } from "react";
import { X } from "lucide-react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { listStoredChatMessages } from "../utils/chat-history-storage";
import type { MessageRecord } from "../types";
import { useChatArtifactsStore } from "../artifacts/store";
import { useResearchRunStore } from "../stores/research-run-store";
import { terminalResearchStatuses } from "../stores/research-run-store";

export const CHAT_WORKSPACE_OPEN_EVENT = "unsloth-open-chat-workspace";

export function openChatWorkspace(): void {
  if (typeof window !== "undefined") {
    window.dispatchEvent(new Event(CHAT_WORKSPACE_OPEN_EVENT));
  }
}

type WorkspaceTab = "outputs" | "background" | "sources" | "subagents";

function messageText(message: MessageRecord): string {
  if (!Array.isArray(message.content)) return "";
  return message.content
    .map((part) => {
      if (!part || typeof part !== "object") return "";
      const value = part as { text?: unknown; content?: unknown };
      return typeof value.text === "string"
        ? value.text
        : typeof value.content === "string"
          ? value.content
          : "";
    })
    .join("\n");
}

function countToolCalls(message: MessageRecord): number {
  if (!Array.isArray(message.content)) return 0;
  return message.content.filter((part) => {
    if (!part || typeof part !== "object") return false;
    const type = (part as { type?: unknown }).type;
    return type === "tool-call" || type === "tool-call-start" || type === "tool-result";
  }).length;
}

export function ChatWorkspacePanel({
  threadId,
  onClose,
}: {
  threadId: string | null;
  onClose: () => void;
}): ReactElement {
  const [tab, setTab] = useState<WorkspaceTab>("outputs");
  const [messages, setMessages] = useState<MessageRecord[]>([]);
  const artifacts = useChatArtifactsStore((state) =>
    Object.values(state.artifactsById).filter(
      (artifact) => !threadId || !artifact.threadId || artifact.threadId === threadId,
    ),
  );
  const openArtifact = useChatArtifactsStore((state) => state.openArtifact);
  const sessions = useResearchRunStore((state) => state.sessions);
  const threadResearch = useMemo(
    () => Object.values(sessions).filter((session) => session.run.threadId === threadId),
    [sessions, threadId],
  );

  useEffect(() => {
    let live = true;
    if (!threadId) {
      setMessages([]);
      return () => {
        live = false;
      };
    }
    void listStoredChatMessages(threadId)
      .then((next) => {
        if (live) setMessages(next);
      })
      .catch(() => {
        if (live) setMessages([]);
      });
    return () => {
      live = false;
    };
  }, [threadId]);

  const sources = useMemo(() => {
    const values = new Set<string>();
    for (const message of messages) {
      for (const url of messageText(message).match(/https?:\/\/[^\s)\]}>,]+/g) ?? []) {
        values.add(url);
      }
    }
    for (const session of threadResearch) {
      for (const source of session.run.sources ?? []) {
        if (source.url) values.add(source.url);
      }
    }
    return [...values];
  }, [messages, threadResearch]);

  const toolCalls = messages.reduce((total, message) => total + countToolCalls(message), 0);
  const activeResearch = threadResearch.filter(
    (session) => !terminalResearchStatuses.has(session.run.status),
  );
  const tabs: Array<{ id: WorkspaceTab; label: string; count: number }> = [
    { id: "outputs", label: "Outputs", count: artifacts.length },
    { id: "background", label: "Background", count: activeResearch.length },
    { id: "sources", label: "Sources", count: sources.length },
    { id: "subagents", label: "Subagents", count: toolCalls },
  ];

  return (
    <section className="flex h-full min-h-0 flex-col border-l border-border/70 bg-background" aria-label="Chat workspace">
      <header className="flex shrink-0 items-center justify-between border-b border-border/70 px-3 py-2">
        <div>
          <h2 className="text-sm font-semibold">Workspace</h2>
          <p className="text-xs text-muted-foreground">Live task outputs and activity</p>
        </div>
        <Button variant="ghost" size="icon-sm" onClick={onClose} aria-label="Close workspace">
          <X className="size-4" />
        </Button>
      </header>
      <nav className="flex shrink-0 gap-1 overflow-x-auto border-b border-border/70 px-2 py-1" aria-label="Workspace tabs">
        {tabs.map((item) => (
          <button
            key={item.id}
            type="button"
            onClick={() => setTab(item.id)}
            className={cn(
              "rounded-md px-2 py-1.5 text-xs text-muted-foreground transition-colors hover:bg-muted hover:text-foreground",
              tab === item.id && "bg-muted font-medium text-foreground",
            )}
            aria-selected={tab === item.id}
            role="tab"
          >
            {item.label}{item.count ? ` · ${item.count}` : ""}
          </button>
        ))}
      </nav>
      <div className="min-h-0 flex-1 overflow-y-auto p-3 text-sm">
        {tab === "outputs" ? (
          artifacts.length ? artifacts.map((artifact) => (
            <button
              key={artifact.id}
              type="button"
              className="mb-2 block w-full rounded-lg border border-border/70 p-3 text-left hover:bg-muted/40"
              onClick={() => openArtifact(artifact, { surface: "panel", view: "preview" })}
            >
              <span className="block font-medium">{artifact.title}</span>
              <span className="text-xs text-muted-foreground">{artifact.source === "tool" ? "Tool output" : "Code output"}</span>
            </button>
          )) : <p className="text-xs text-muted-foreground">Outputs appear here when the agent creates an artifact.</p>
        ) : null}
        {tab === "background" ? (
          activeResearch.length ? activeResearch.map((session) => (
            <div key={session.run.id} className="mb-2 rounded-lg border border-border/70 p-3">
              <p className="font-medium">Research activity</p>
              <p className="text-xs text-muted-foreground">{session.run.status} · {session.activities.length} events</p>
            </div>
          )) : <p className="text-xs text-muted-foreground">No active background process. Research and training continue here when started.</p>
        ) : null}
        {tab === "sources" ? (
          sources.length ? sources.map((source) => (
            <a key={source} href={source} target="_blank" rel="noreferrer" className="mb-2 block break-all rounded-lg border border-border/70 p-2 text-xs text-primary hover:underline">{source}</a>
          )) : <p className="text-xs text-muted-foreground">Sources cited by this task will appear here.</p>
        ) : null}
        {tab === "subagents" ? (
          <div className="rounded-lg border border-border/70 p-3">
            <p className="font-medium">Local agent activity</p>
            <p className="mt-1 text-xs text-muted-foreground">{toolCalls} tool events recorded in this task. Parallel research activity is shown in Background.</p>
            <p className="mt-2 text-xs text-muted-foreground">Subagent execution stays local to this harness and shares the task’s permissions.</p>
          </div>
        ) : null}
      </div>
    </section>
  );
}
