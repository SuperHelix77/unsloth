// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";

export type MemorySearchResult = {
  id?: string;
  memory?: string;
  score?: number;
  metadata?: Record<string, unknown>;
};

export type MemorySearchResponse = {
  results: MemorySearchResult[];
  available: boolean;
  reason?: string;
};

async function jsonOrThrow<T>(response: Response): Promise<T> {
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = (body as { detail?: string } | null)?.detail;
    throw new Error(detail ?? `Request failed (${response.status})`);
  }
  return body as T;
}

export async function searchLearningMemory(
  query: string,
  limit = 5,
): Promise<MemorySearchResponse> {
  return jsonOrThrow(
    await authFetch("/api/memory/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, limit }),
    }),
  );
}
