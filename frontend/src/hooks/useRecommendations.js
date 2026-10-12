/**
 * src/hooks/useRecommendations.js
 * ================================
 * Hook for Phase 3 outfit + missing-item recommendations.
 */

import { useState, useCallback } from 'react';
import { recommendOutfits, recommendMissingItem } from '../api/recommendationApi.js';

export function useOutfitRecommendations() {
  const [results, setResults]   = useState(null);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState(null);
  const [modelNote, setModelNote] = useState(null);

  const generate = useCallback(async (selectedItems, topK = 5) => {
    setLoading(true);
    setError(null);
    setResults(null);
    try {
      const data = await recommendOutfits(selectedItems, topK);
      setResults(data.recommendations || []);
      setModelNote(data.model_note || null);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setResults(null);
    setError(null);
    setModelNote(null);
  }, []);

  return { results, loading, error, modelNote, generate, reset };
}

export function useMissingItemRecommendation() {
  const [results, setResults]   = useState(null);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState(null);
  const [modelNote, setModelNote] = useState(null);

  const find = useCallback(async (outfitItems, candidateItems, topK = 5) => {
    setLoading(true);
    setError(null);
    setResults(null);
    try {
      const data = await recommendMissingItem(outfitItems, candidateItems, topK);
      setResults(data.recommendations || []);
      setModelNote(data.model_note || null);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setResults(null);
    setError(null);
    setModelNote(null);
  }, []);

  return { results, loading, error, modelNote, find, reset };
}
