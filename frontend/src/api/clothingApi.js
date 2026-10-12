/**
 * src/api/clothingApi.js
 * ======================
 * Phase 1 + Phase 2 API calls.
 *
 * Exact endpoint contracts from api/main.py:
 *
 * POST /analyze-clothing
 *   field: file (UploadFile)
 *   query: top_k=3, n_colors=3
 *   returns: { category, confidence, top_predictions, dominant_color,
 *              dominant_hex, color_palette, color_distribution, attributes }
 *
 * POST /similar-clothing
 *   field: file (UploadFile)
 *   query: top_k=5
 *   returns: { query_category, query_confidence, results, index_size }
 *
 * GET /health
 *   returns: { status, model_loaded, embeddings_loaded,
 *              similarity_index_loaded, compatibility_model_loaded,
 *              index_size, num_classes, classes }
 */

import { apiGet, apiPostForm } from './client.js';

/**
 * Analyze a clothing image. Returns full Phase 2 profile.
 * @param {File} file
 * @returns {Promise<AnalyzeResponse>}
 */
export async function analyzeClothing(file) {
  const form = new FormData();
  form.append('file', file);
  return apiPostForm('/analyze-clothing?top_k=3&n_colors=3', form);
}

/**
 * Find visually similar clothing items.
 * @param {File} file
 * @param {number} topK
 * @returns {Promise<SimilarResponse>}
 */
export async function findSimilarClothing(file, topK = 5) {
  const form = new FormData();
  form.append('file', file);
  return apiPostForm(`/similar-clothing?top_k=${topK}`, form);
}

/**
 * Get API health status.
 * @returns {Promise<HealthResponse>}
 */
export async function getHealth() {
  return apiGet('/health');
}
