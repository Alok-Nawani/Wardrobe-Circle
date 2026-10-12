/**
 * src/pages/GenerateOutfit.jsx
 * =============================
 * Style Studio page. Select items from a beautiful gallery and generate looks.
 */

import { useState } from 'react';
import { useWardrobeContext } from '../context/WardrobeContext.jsx';
import { useOutfitRecommendations } from '../hooks/useRecommendations.js';
import ClothingCard from '../components/ClothingCard.jsx';
import OutfitCard from '../components/OutfitCard.jsx';
import ModelNote from '../components/ModelNote.jsx';
import { PageLoading } from '../components/ui/LoadingSpinner.jsx';

const VALID_CATEGORIES = new Set(['Tops','Bottoms','Dresses','Outerwear','Footwear','Accessories']);

export default function GenerateOutfit() {
  const { items, loading: wardrobeLoading } = useWardrobeContext();
  const { results, loading, error, modelNote, generate, reset } = useOutfitRecommendations();
  const [selected, setSelected] = useState(new Set());

  const eligibleItems = items.filter(it => VALID_CATEGORIES.has(it.category));
  const selectedItems = eligibleItems.filter(it => selected.has(it.id));

  const toggleItem = (item) => {
    setSelected(prev => {
      const next = new Set(prev);
      if (next.has(item.id)) next.delete(item.id);
      else next.add(item.id);
      return next;
    });
    reset(); // Clear old results when selection changes
  };

  const handleGenerate = () => {
    if (selectedItems.length < 2) return;
    generate(selectedItems, 5);
  };

  if (wardrobeLoading) return <PageLoading message="Opening Style Studio..." />;

  return (
    <div className="flex flex-col gap-16 animate-fade-in">
      
      <section className="text-center max-w-2xl mx-auto space-y-4">
        <p className="editorial-subheading">Style Studio</p>
        <h1 className="editorial-heading">Create your look</h1>
        <p className="text-brand-subtext">Select pieces from your wardrobe. The AI will curate compatible combinations.</p>
      </section>

      {/* ── Selection Area ── */}
      <section className="bg-[#EAE7DF] p-8 md:p-12 -mx-6 md:-mx-12 flex flex-col gap-8">
        
        <div className="flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            <span className="text-sm font-semibold uppercase tracking-widest">{selected.size} Selected</span>
            {selected.size > 0 && (
              <button onClick={() => { setSelected(new Set()); reset(); }} className="text-xs text-brand-subtext uppercase tracking-widest hover:text-brand-text transition-colors">
                Clear
              </button>
            )}
          </div>
          
          <button
            onClick={handleGenerate}
            disabled={selectedItems.length < 2 || loading}
            className="btn-primary bg-brand-text shadow-lg disabled:opacity-50"
          >
            {loading ? 'Curating...' : 'Generate Look'}
          </button>
        </div>

        {error && <div className="text-red-600 text-sm">{error}</div>}
        {selectedItems.length > 0 && selectedItems.length < 2 && (
          <p className="text-xs text-brand-accent uppercase tracking-widest">Select at least 2 pieces</p>
        )}

        <div className="flex gap-4 overflow-x-auto pb-4 no-scrollbar">
          {eligibleItems.map(item => (
            <div key={item.id} className="w-48 md:w-56 shrink-0">
              <ClothingCard
                item={item}
                selectable
                selected={selected.has(item.id)}
                onSelect={toggleItem}
              />
            </div>
          ))}
        </div>
      </section>

      {/* ── Results Area ── */}
      {results !== null && (
        <section className="flex flex-col gap-8 max-w-5xl mx-auto w-full animate-slide-up">
          <h2 className="font-serif text-3xl md:text-4xl text-center mb-4">
            {results.length === 0 ? 'No compatible looks found.' : 'Your Looks'}
          </h2>

          {modelNote && (
            <div className="text-center mb-8">
              <ModelNote note={modelNote} />
            </div>
          )}

          {results.length === 0 && (
            <p className="text-center text-brand-subtext">Try selecting different items or adding more variety to your wardrobe.</p>
          )}

          <div className="flex flex-col">
            {results.map((outfit, i) => (
              <OutfitCard key={i} outfit={outfit} rank={i} />
            ))}
          </div>
        </section>
      )}

    </div>
  );
}
