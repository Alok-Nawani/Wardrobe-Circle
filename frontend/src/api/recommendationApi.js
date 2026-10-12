/**
 * src/api/recommendationApi.js
 * ============================
 * Phase 3 recommendation API calls.
 *
 * Exact endpoint contracts from api/main.py:
 *
 * POST /recommend-outfits
 *   body: { items: WardrobeItemIn[], top_k: int }
 *   WardrobeItemIn: { item_id, category, image_path?, dominant_color?,
 *                     dominant_hex?, metadata? }
 *   returns: { recommendations: OutfitResult[], model_note: str }
 *   OutfitResult: { items: dict[], compatibility_score: float, categories: str[] }
 *
 * POST /recommend-missing-item
 *   body: { outfit_items: WardrobeItemIn[], candidate_items: WardrobeItemIn[], top_k: int }
 *   returns: { recommendations: MissingItemResult[], model_note: str }
 *   MissingItemResult: { item: dict, compatibility_score: float }
 */

import { apiPost } from './client.js';

/**
 * Convert a wardrobe item record to WardrobeItemIn schema.
 * @param {Object} item - wardrobe item from /wardrobe/items
 * @returns {Object} WardrobeItemIn
 */
function toItemIn(item) {
  return {
    item_id:        item.id,
    category:       item.category,
    image_path:     item.image_url || null,
    dominant_color: item.dominant_color || null,
    dominant_hex:   item.dominant_hex || null,
    metadata:       {},
  };
}

/**
 * Get outfit recommendations for selected wardrobe items.
 * @param {Object[]} items - wardrobe items (from /wardrobe/items)
 * @param {number} topK
 * @returns {Promise<RecommendOutfitsResponse>}
 */
export async function recommendOutfits(items, topK = 5) {
  return apiPost('/recommend-outfits', {
    items: items.map(toItemIn),
    top_k: topK,
  });
}

/**
 * Rank candidate items to fill a gap in a partial outfit.
 * @param {Object[]} outfitItems  - items already in the outfit
 * @param {Object[]} candidateItems - items to rank
 * @param {number} topK
 * @returns {Promise<RecommendMissingResponse>}
 */
export async function recommendMissingItem(outfitItems, candidateItems, topK = 5) {
  return apiPost('/recommend-missing-item', {
    outfit_items:    outfitItems.map(toItemIn),
    candidate_items: candidateItems.map(toItemIn),
    top_k:           topK,
  });
}

/**
 * ShoppingProvider stub — for future online store integration.
 * When a real shopping API is connected, implement this interface.
 *
 * searchProducts({ category, color, attributes }) → Product[]
 *
 * Currently returns null to indicate no provider is configured.
 * The UI shows "Online shopping search is not connected yet."
 */
export const ShoppingProvider = {
  isConnected: false,
  async searchProducts({ category, color, attributes } = {}) {
    // Future: integrate with a real shopping API here.
    // DO NOT add fake product data.
    return null;
  },
};
