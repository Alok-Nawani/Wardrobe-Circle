/**
 * src/components/OutfitCard.jsx
 * ==============================
 * Displays an outfit recommendation as an editorial lookbook composition.
 */

import { imageUrl } from '../api/client.js';

function CompositionItem({ item }) {
  return (
    <div className="flex flex-col gap-2">
      <div className="clothing-image-container aspect-[3/4]">
        {item.image_url ? (
          <img
            src={imageUrl(item.image_url)}
            alt={item.category}
            className="clothing-image"
            loading="lazy"
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center text-brand-subtext text-xs uppercase tracking-widest font-serif">
            No Image
          </div>
        )}
      </div>
      <div className="flex justify-between items-center px-1">
        <span className="text-[10px] uppercase tracking-widest font-semibold text-brand-text">
          {item.category}
        </span>
        {item.dominant_hex && (
          <div
            className="w-2.5 h-2.5 rounded-full border border-brand-border"
            style={{ backgroundColor: item.dominant_hex }}
            title={item.dominant_color}
          />
        )}
      </div>
    </div>
  );
}

export default function OutfitCard({ outfit, rank }) {
  const { items, compatibility_score } = outfit;

  return (
    <div className="border-b border-brand-border pb-16 pt-8 animate-slide-up" style={{ animationDelay: `${rank * 100}ms` }}>
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between mb-8 gap-4">
        <div>
          <span className="text-xs font-bold text-brand-subtext/50 tabular-nums uppercase tracking-widest block mb-1">
            Look {String(rank + 1).padStart(2, '0')}
          </span>
          <h3 className="font-serif text-2xl text-brand-text">Recommended Combination</h3>
        </div>
        
        <div className="flex items-center gap-3">
          <span className="text-[10px] uppercase tracking-widest text-brand-subtext">Compatibility</span>
          <span className="font-serif text-2xl text-brand-accent">
            {Math.round(compatibility_score * 100)}%
          </span>
        </div>
      </div>

      {/* Composition Gallery */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {items.map((item, i) => (
          <CompositionItem key={item.item_id || i} item={item} />
        ))}
      </div>
    </div>
  );
}
