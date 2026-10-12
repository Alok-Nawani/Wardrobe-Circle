/**
 * src/api/wardrobeApi.js
 * ======================
 * Wardrobe CRUD API calls.
 *
 * Exact endpoint contracts from api/main.py:
 *
 * POST /wardrobe/items           (multipart)  → item record
 * GET  /wardrobe/items           → { items: [...], count: n }
 * GET  /wardrobe/items/{id}      → item record
 * DELETE /wardrobe/items/{id}    → { deleted: id }
 *
 * Item record shape:
 * {
 *   id, category, confidence, dominant_color, dominant_hex,
 *   color_palette, color_distribution, attributes,
 *   image_url, created_at
 * }
 */

import { apiGet, apiPostForm, apiDelete } from './client.js';

/**
 * Upload an image, analyze it, and save to wardrobe.
 * @param {File} file
 * @returns {Promise<WardrobeItem>}
 */
export async function addWardrobeItem(file) {
  const form = new FormData();
  form.append('file', file);
  return apiPostForm('/wardrobe/items', form);
}

/**
 * List all wardrobe items (newest first).
 * @returns {Promise<{items: WardrobeItem[], count: number}>}
 */
export async function listWardrobeItems() {
  return apiGet('/wardrobe/items');
}

/**
 * Get a single wardrobe item by ID.
 * @param {string} id
 * @returns {Promise<WardrobeItem>}
 */
export async function getWardrobeItem(id) {
  return apiGet(`/wardrobe/items/${id}`);
}

/**
 * Delete a wardrobe item.
 * @param {string} id
 * @returns {Promise<{deleted: string}>}
 */
export async function deleteWardrobeItem(id) {
  return apiDelete(`/wardrobe/items/${id}`);
}
