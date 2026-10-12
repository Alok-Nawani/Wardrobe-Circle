/**
 * src/context/WardrobeContext.jsx
 * ================================
 * Global wardrobe state — list of items fetched from the backend.
 * Provides loading, error, and CRUD helpers to all pages.
 */

import { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { listWardrobeItems, deleteWardrobeItem } from '../api/wardrobeApi.js';

const WardrobeContext = createContext(null);

export function WardrobeProvider({ children }) {
  const [items, setItems]     = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState(null);

  const fetchItems = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await listWardrobeItems();
      setItems(data.items || []);
    } catch (e) {
      setError(e.message);
      setItems([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchItems(); }, [fetchItems]);

  const addItem = useCallback((newItem) => {
    setItems(prev => [newItem, ...prev]);
  }, []);

  const removeItem = useCallback(async (id) => {
    try {
      await deleteWardrobeItem(id);
      setItems(prev => prev.filter(it => it.id !== id));
      return true;
    } catch (e) {
      return false;
    }
  }, []);

  return (
    <WardrobeContext.Provider value={{ items, loading, error, fetchItems, addItem, removeItem }}>
      {children}
    </WardrobeContext.Provider>
  );
}

export function useWardrobeContext() {
  const ctx = useContext(WardrobeContext);
  if (!ctx) throw new Error('useWardrobeContext must be inside WardrobeProvider');
  return ctx;
}
