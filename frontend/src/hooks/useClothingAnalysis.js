/**
 * src/hooks/useClothingAnalysis.js
 * =================================
 * Hook for managing clothing analysis state.
 * Wraps analyzeClothing + findSimilarClothing API calls.
 */

import { useState, useCallback } from 'react';
import { analyzeClothing, findSimilarClothing } from '../api/clothingApi.js';

export default function useClothingAnalysis() {
  const [file, setFile]             = useState(null);
  const [preview, setPreview]       = useState(null);
  const [analysis, setAnalysis]     = useState(null);
  const [similar, setSimilar]       = useState(null);
  const [loading, setLoading]       = useState(false);
  const [loadingSimilar, setLoadingSimilar] = useState(false);
  const [error, setError]           = useState(null);

  const selectFile = useCallback((selectedFile) => {
    setFile(selectedFile);
    setAnalysis(null);
    setSimilar(null);
    setError(null);
    const url = URL.createObjectURL(selectedFile);
    setPreview(url);
  }, []);

  const analyze = useCallback(async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setAnalysis(null);
    setSimilar(null);
    try {
      const result = await analyzeClothing(file);
      setAnalysis(result);
      // Fetch similar items in background
      setLoadingSimilar(true);
      try {
        const sim = await findSimilarClothing(file, 5);
        setSimilar(sim);
      } catch {
        setSimilar(null); // non-fatal
      } finally {
        setLoadingSimilar(false);
      }
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [file]);

  const reset = useCallback(() => {
    setFile(null);
    setPreview(null);
    setAnalysis(null);
    setSimilar(null);
    setError(null);
    setLoading(false);
  }, []);

  return { file, preview, analysis, similar, loading, loadingSimilar, error, selectFile, analyze, reset };
}
