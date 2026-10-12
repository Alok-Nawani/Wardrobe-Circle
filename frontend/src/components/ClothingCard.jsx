/**
 * src/components/ClothingCard.jsx
 * ================================
 * Editorial clothing card. Large images, object-fit: contain, subtle metadata.
 */

import { useState } from 'react';
import { Trash2 } from 'lucide-react';
import { imageUrl } from '../api/client.js';

export default function ClothingCard({ item, onDelete, selectable = false, selected = false, onSelect }) {
  const [deleting, setDeleting] = useState(false);
  const [imgError, setImgError] = useState(false);

  const handleDelete = async (e) => {
    e.stopPropagation();
    if (!confirm('Remove this item from your wardrobe?')) return;
    setDeleting(true);
    const ok = await onDelete(item.id);
    if (!ok) setDeleting(false);
  };

  const handleClick = () => {
    if (selectable && onSelect) onSelect(item);
  };

  return (
    <div
      className={`group relative flex flex-col cursor-${selectable ? 'pointer' : 'default'} transition-all duration-300 ease-editorial ${deleting ? 'opacity-40 pointer-events-none' : ''}`}
      onClick={handleClick}
      role={selectable ? 'checkbox' : undefined}
      aria-checked={selectable ? selected : undefined}
      tabIndex={selectable ? 0 : undefined}
      onKeyDown={selectable ? (e) => { if (e.key === ' ' || e.key === 'Enter') handleClick(); } : undefined}
    >
      {/* Image Container */}
      <div className={`clothing-image-container aspect-[3/4] mb-4 ${selected ? 'ring-1 ring-brand-accent ring-offset-2 ring-offset-brand-bg' : ''}`}>
        {!imgError && item.image_url ? (
          <img
            src={imageUrl(item.image_url)}
            alt={`${item.category} item`}
            className="clothing-image"
            loading="lazy"
            onError={() => setImgError(true)}
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center text-brand-subtext text-xl font-serif">
            No Image
          </div>
        )}

        {/* Selection Indicator */}
        {selectable && (
          <div className={`absolute top-4 left-4 z-10 w-5 h-5 rounded-full border flex items-center justify-center transition-all duration-300 ease-editorial
            ${selected ? 'bg-brand-accent border-brand-accent' : 'bg-transparent border-brand-subtext/50 group-hover:border-brand-accent'}
          `}>
            {selected && <div className="w-2 h-2 bg-white rounded-full" />}
          </div>
        )}

        {/* Delete Button */}
        {onDelete && !selectable && (
          <button
            onClick={handleDelete}
            className="absolute top-4 right-4 z-10 w-8 h-8 rounded-full bg-white/80 backdrop-blur-sm flex items-center justify-center
              opacity-0 group-hover:opacity-100 transition-all duration-300 ease-editorial hover:bg-white hover:text-red-600 text-brand-text shadow-sm"
            aria-label={`Delete ${item.category}`}
          >
            <Trash2 size={14} strokeWidth={1.5} />
          </button>
        )}
      </div>

      {/* Metadata */}
      <div className="flex flex-col gap-1 px-1">
        <h3 className="text-sm font-semibold text-brand-text tracking-wide uppercase">{item.category}</h3>
        <div className="flex items-center justify-between">
          <p className="text-xs text-brand-subtext capitalize">{item.dominant_color || 'Unknown color'}</p>
          {item.confidence !== null && (
            <span className="text-[10px] text-brand-subtext/70 uppercase tracking-wider">
              {Math.round(item.confidence * 100)}% match
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
