/**
 * src/pages/FindMissingItem.jsx
 * ==============================
 * Missing item page acting as an AI Stylist.
 */

import { useState } from 'react';
import { useWardrobeContext } from '../context/WardrobeContext.jsx';
import { useMissingItemRecommendation } from '../hooks/useRecommendations.js';
import { ShoppingProvider } from '../api/recommendationApi.js';
import ClothingCard from '../components/ClothingCard.jsx';
import ModelNote from '../components/ModelNote.jsx';
import { PageLoading } from '../components/ui/LoadingSpinner.jsx';
import { imageUrl } from '../api/client.js';

const VALID_CATEGORIES = new Set(['Tops','Bottoms','Dresses','Outerwear','Footwear','Accessories']);

function MissingItemResult({ rec, rank }) {
  const item = rec.item;
  return (
    <div className="flex flex-col gap-4 animate-slide-up" style={{ animationDelay: `${rank * 100}ms` }}>
      <div className="clothing-image-container aspect-[3/4]">
        {item.image_url ? (
          <img
            src={imageUrl(item.image_url)}
            alt={item.category}
            className="clothing-image"
            loading="lazy"
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center text-brand-subtext font-serif">No Image</div>
        )}
      </div>
      
      <div className="flex flex-col gap-1 px-1">
        <div className="flex items-center justify-between">
          <span className="text-xs font-bold text-brand-subtext/50 tabular-nums uppercase tracking-widest">#{rank + 1}</span>
          <span className="text-[10px] uppercase tracking-widest text-brand-accent font-semibold">
            {Math.round(rec.compatibility_score * 100)}% Match
          </span>
        </div>
        <h3 className="text-sm font-semibold text-brand-text tracking-wide uppercase">{item.category}</h3>
        {item.dominant_color && (
          <p className="text-xs text-brand-subtext capitalize">{item.dominant_color}</p>
        )}
      </div>
    </div>
  );
}

export default function FindMissingItem() {
  const { items, loading: wardrobeLoading } = useWardrobeContext();
  const { results, loading, error, modelNote, find, reset } = useMissingItemRecommendation();

  const [outfitIds, setOutfitIds] = useState(new Set());
  const [candidateIds, setCandidateIds] = useState(new Set());

  const eligible = items.filter(it => VALID_CATEGORIES.has(it.category));
  const outfitItems = eligible.filter(it => outfitIds.has(it.id));
  const candidateItems = eligible.filter(it => candidateIds.has(it.id));

  const toggleOutfit = (item) => {
    setOutfitIds(prev => { const n = new Set(prev); if (n.has(item.id)) n.delete(item.id); else n.add(item.id); return n; });
    reset();
  };

  const toggleCandidate = (item) => {
    setCandidateIds(prev => { const n = new Set(prev); if (n.has(item.id)) n.delete(item.id); else n.add(item.id); return n; });
    reset();
  };

  const handleFind = () => {
    if (candidateItems.length === 0) return;
    find(outfitItems, candidateItems, 4);
  };

  if (wardrobeLoading) return <PageLoading message="Consulting stylist..." />;

  return (
    <div className="flex flex-col gap-16 animate-fade-in">
      
      <section className="text-center max-w-2xl mx-auto space-y-4">
        <p className="editorial-subheading">Stylist</p>
        <h1 className="editorial-heading">Complete your look</h1>
        <p className="text-brand-subtext">Select pieces you are wearing, and candidate pieces to add. The AI ranks the best match.</p>
      </section>

      {/* ── Step 1: Outfit ── */}
      <section>
        <div className="flex justify-between items-end mb-6">
          <h2 className="section-title !mb-0">1. Your Look</h2>
          {outfitIds.size > 0 && (
            <button onClick={() => { setOutfitIds(new Set()); reset(); }} className="text-xs uppercase tracking-widest text-brand-subtext">Clear</button>
          )}
        </div>
        <div className="flex gap-4 overflow-x-auto pb-4 no-scrollbar">
          {eligible.map(item => (
            <div key={item.id} className="w-32 md:w-40 shrink-0">
              <ClothingCard item={item} selectable selected={outfitIds.has(item.id)} onSelect={toggleOutfit} />
            </div>
          ))}
        </div>
      </section>

      {/* ── Step 2: Candidates ── */}
      <section>
        <div className="flex justify-between items-end mb-6">
          <h2 className="section-title !mb-0">2. Candidates</h2>
          {candidateIds.size > 0 && (
            <button onClick={() => { setCandidateIds(new Set()); reset(); }} className="text-xs uppercase tracking-widest text-brand-subtext">Clear</button>
          )}
        </div>
        <div className="flex gap-4 overflow-x-auto pb-4 no-scrollbar">
          {eligible.map(item => (
            <div key={item.id} className="w-32 md:w-40 shrink-0">
              <ClothingCard item={item} selectable selected={candidateIds.has(item.id)} onSelect={toggleCandidate} />
            </div>
          ))}
        </div>
      </section>

      {/* ── Action ── */}
      <section className="flex flex-col items-center gap-4">
        <button
          onClick={handleFind}
          disabled={candidateItems.length === 0 || loading}
          className="btn-primary px-12"
        >
          {loading ? 'Finding match...' : 'Discover Best Match'}
        </button>
        {candidateItems.length === 0 && (
          <p className="text-xs uppercase tracking-widest text-brand-accent">Select at least 1 candidate</p>
        )}
        {error && <p className="text-sm text-red-600">{error}</p>}
      </section>

      {/* ── Results ── */}
      {results !== null && (
        <section className="pt-12 border-t border-brand-border animate-slide-up flex flex-col gap-10">
          <h2 className="font-serif text-3xl md:text-4xl text-center">
            {results.length === 0 ? 'No recommendations available' : 'Recommended Options'}
          </h2>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {results.map((rec, i) => (
              <MissingItemResult key={i} rec={rec} rank={i} />
            ))}
          </div>

          {modelNote && (
            <div className="text-center mt-4">
              <ModelNote note={modelNote} />
            </div>
          )}

          {!ShoppingProvider.isConnected && results.length > 0 && (
            <div className="mt-8 text-center p-8 bg-[#EAE7DF]">
              <p className="text-sm uppercase tracking-widest font-semibold mb-2">Looking for something you don't own?</p>
              <p className="text-sm text-brand-subtext max-w-md mx-auto leading-relaxed">
                Online shopping integration can be connected here later.
              </p>
            </div>
          )}
        </section>
      )}

    </div>
  );
}
