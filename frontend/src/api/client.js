/**
 * src/api/client.js
 * =================
 * Centralised fetch wrapper for all Wardrobe Circle API calls.
 *
 * - Uses VITE_API_BASE_URL (never hardcoded localhost).
 * - Throws ApiError with structured message on non-2xx responses.
 * - Handles network failures separately.
 */

const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export class ApiError extends Error {
  constructor(message, status, detail) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

async function handleResponse(res) {
  if (res.ok) {
    // 204 No Content
    if (res.status === 204) return null;
    return res.json();
  }

  let detail = `HTTP ${res.status}`;
  try {
    const body = await res.json();
    detail = body.detail || JSON.stringify(body);
  } catch {
    // non-JSON error body
  }

  throw new ApiError(
    `Request failed: ${detail}`,
    res.status,
    detail
  );
}

export async function apiGet(path) {
  try {
    const res = await fetch(`${BASE_URL}${path}`);
    return handleResponse(res);
  } catch (e) {
    if (e instanceof ApiError) throw e;
    throw new ApiError(
      'Unable to reach the Wardrobe Circle server. Make sure FastAPI is running.',
      0,
      e.message
    );
  }
}

export async function apiPost(path, body) {
  try {
    const res = await fetch(`${BASE_URL}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    return handleResponse(res);
  } catch (e) {
    if (e instanceof ApiError) throw e;
    throw new ApiError(
      'Unable to reach the Wardrobe Circle server. Make sure FastAPI is running.',
      0,
      e.message
    );
  }
}

export async function apiPostForm(path, formData) {
  try {
    const res = await fetch(`${BASE_URL}${path}`, {
      method: 'POST',
      body: formData,   // multipart/form-data — do NOT set Content-Type header
    });
    return handleResponse(res);
  } catch (e) {
    if (e instanceof ApiError) throw e;
    throw new ApiError(
      'Unable to reach the Wardrobe Circle server. Make sure FastAPI is running.',
      0,
      e.message
    );
  }
}

export async function apiDelete(path) {
  try {
    const res = await fetch(`${BASE_URL}${path}`, { method: 'DELETE' });
    return handleResponse(res);
  } catch (e) {
    if (e instanceof ApiError) throw e;
    throw new ApiError(
      'Unable to reach the Wardrobe Circle server. Make sure FastAPI is running.',
      0,
      e.message
    );
  }
}

/** Returns the full URL for a wardrobe image path */
export function imageUrl(relativePath) {
  if (!relativePath) return null;
  if (relativePath.startsWith('http')) return relativePath;
  return `${BASE_URL}${relativePath}`;
}
