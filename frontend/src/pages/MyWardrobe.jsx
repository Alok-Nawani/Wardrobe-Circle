/**
 * src/pages/MyWardrobe.jsx
 * =========================
 * Premium digital wardrobe list. Large images, minimal filters.
 */

import { useState, useMemo } from 'react';
import { Search } from 'lucide-react';
import { useWardrobeContext } from '../context/WardrobeContext.jsx';
import ClothingCard from '../components/ClothingCard.jsx';
import { PageLoading } from '../components/ui/LoadingSpinner.jsx';
import ErrorState from '../components/ui/ErrorState.jsx';
import { Toast, useToast } from '../components/ui/Toast.jsx';

const CATEGORIES = ['All', 'Tops', 'Bottoms', 'Outerwear', 'Dresses', 'Footwear', 'Accessories'];

export default function MyWardrobe() {
  const { items, loading, error, fetchItems, removeItem } = useWardrobeContext();
  const { toast, show, hide } = useToast();

  const [activeCategory, setActiveCategory] = useState('All');
  const [query, setQuery] = useState('');

  const filtered = useMemo(() => {
    let result = [...items];
    if (activeCategory !== 'All') {
      result = result.filter(it => it.category === activeCategory);
    }
    if (query.trim()) {
      const q = query.toLowerCase();
      result = result.filter(it =>
        it.category?.toLowerCase().includes(q) ||
        it.dominant_color?.toLowerCase().includes(q)
      );
    }
    // Default sort: newest
    result.sort((a, b) => b.created_at.localeCompare(a.created_at));
    return result;
  }, [items, activeCategory, query]);

  const handleDelete = async (id) => {
    const ok = await removeItem(id);
    if (ok) show('Item removed from collection');
    else show('Failed to remove item', 'error');
    return ok;
  };

  if (loading) return <PageLoading message="Curating your wardrobe..." />;
  if (error) return <div className="py-20"><ErrorState message={error} onRetry={fetchItems} /></div>;

  return (
    <div className="flex flex-col gap-12 animate-fade-in">
      {toast && <Toast message={toast.message} type={toast.type} onClose={hide} />}

      {/* ── Header ── */}
      <section className="flex flex-col items-center text-center gap-4 max-w-2xl mx-auto">
        <p className="editorial-subheading">Collection</p>
        <h1 className="editorial-heading">Everything you own,<br/><span className="italic text-brand-subtext">beautifully organized.</span></h1>
        <p className="text-sm text-brand-subtext mt-2 uppercase tracking-widest">{items.length} Pieces</p>
      </section>

      {items.length === 0 ? (
        <section className="w-full bg-[#EAE7DF] p-16 flex items-center justify-center text-center mt-8">
          <div className="space-y-6">
            <h2 className="font-serif text-3xl text-brand-accent">Your collection is empty.</h2>
            <div className="pt-4">
              <a href="/add" className="btn-primary">Add a Piece</a>
            </div>
          </div>
        </section>
      ) : (
        <>
          {/* ── Filters ── */}
          <section className="flex flex-col md:flex-row items-center justify-between gap-6 border-b border-brand-border pb-6 sticky top-20 bg-brand-bg/95 backdrop-blur-md z-30">
            <div className="flex gap-2 overflow-x-auto w-full md:w-auto no-scrollbar pb-2 md:pb-0">
              {CATEGORIES.map(cat => {
                const count = items.filter(it => it.category === cat).length;
                if (cat !== 'All' && count === 0) return null; // hide empty categories
                return (
                  <button
                    key={cat}
                    onClick={() => setActiveCategory(cat)}
                    className={`pill-filter whitespace-nowrap ${activeCategory === cat ? 'pill-filter-active' : 'pill-filter-inactive'}`}
                  >
                    {cat} <span className="opacity-60 ml-1 text-[10px]">{cat !== 'All' ? count : ''}</span>
                  </button>
                );
              })}
            </div>
            <div className="relative w-full md:w-64">
              <Search size={16} className="absolute left-4 top-1/2 -translate-y-1/2 text-brand-subtext" strokeWidth={1.5} />
              <input
                type="text"
                placeholder="Search collection..."
                value={query}
                onChange={e => setQuery(e.target.value)}
                className="w-full bg-transparent border border-brand-border px-10 py-2.5 text-sm outline-none focus:border-brand-text transition-colors rounded-none placeholder:text-brand-subtext"
              />
            </div>
          </section>

          {/* ── Grid ── */}
          {filtered.length === 0 ? (
            <div className="text-center py-20 text-brand-subtext tracking-wide uppercase text-sm">
              No pieces match your search.
            </div>
          ) : (
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-x-6 gap-y-12">
              {filtered.map(item => (
                <ClothingCard key={item.id} item={item} onDelete={handleDelete} />
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
