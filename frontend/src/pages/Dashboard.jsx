/**
 * src/pages/Dashboard.jsx
 * ========================
 * Premium editorial homepage. No generic metrics.
 * Focuses on visual presentation of the wardrobe and core actions.
 */

import { Link } from 'react-router-dom';
import { ArrowRight, Sparkles, Search } from 'lucide-react';
import { useWardrobeContext } from '../context/WardrobeContext.jsx';
import ClothingCard from '../components/ClothingCard.jsx';
import { PageLoading } from '../components/ui/LoadingSpinner.jsx';
import ErrorState from '../components/ui/ErrorState.jsx';

export default function Dashboard() {
  const { items, loading, error, fetchItems, removeItem } = useWardrobeContext();

  const hour = new Date().getHours();
  const greeting = hour < 12 ? 'Good morning' : hour < 17 ? 'Good afternoon' : 'Good evening';

  if (loading) return <PageLoading message="Loading your wardrobe" />;
  if (error)   return (
    <div className="py-20">
      <ErrorState message={error} onRetry={fetchItems} />
    </div>
  );

  const recentItems = items.slice(0, 4);
  const hasItems = items.length > 0;

  return (
    <div className="flex flex-col gap-24 md:gap-32 animate-fade-in pb-12">
      
      {/* ── Hero ── */}
      <section className="flex flex-col gap-6 md:gap-8 max-w-3xl">
        <p className="editorial-subheading text-brand-accent/70">{greeting}</p>
        <h1 className="editorial-heading">
          Your wardrobe isn't just a closet.<br />
          <span className="italic text-brand-subtext">It's a world.</span>
        </h1>
        
        {!hasItems ? (
          <div className="flex flex-col sm:flex-row gap-4 mt-4">
            <Link to="/add" className="btn-primary">Add Your First Piece</Link>
          </div>
        ) : (
          <div className="mt-4 flex items-center gap-6">
            <Link to="/wardrobe" className="btn-primary">View Collection ({items.length})</Link>
            <Link to="/add" className="text-sm uppercase tracking-widest font-semibold hover:text-brand-subtext transition-colors">Add Item +</Link>
          </div>
        )}
      </section>

      {/* ── Visual Wardrobe (Recent) ── */}
      {hasItems && (
        <section>
          <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 mb-10">
            <div>
              <h2 className="section-title mb-2">Recently Added</h2>
              <p className="text-sm text-brand-subtext max-w-md leading-relaxed">
                The latest pieces curated and analyzed for your digital collection.
              </p>
            </div>
            <Link to="/wardrobe" className="group flex items-center gap-2 text-xs uppercase tracking-widest font-semibold">
              Explore All
              <ArrowRight size={14} className="transition-transform group-hover:translate-x-1" />
            </Link>
          </div>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 md:gap-8">
            {recentItems.map((item, i) => (
              <div key={item.id} className="animate-slide-up" style={{ animationDelay: `${i * 100}ms` }}>
                <ClothingCard item={item} onDelete={removeItem} />
              </div>
            ))}
          </div>
        </section>
      )}

      {/* ── Empty State Editorial ── */}
      {!hasItems && (
        <section className="w-full bg-[#EAE7DF] p-12 md:p-24 flex items-center justify-center text-center">
          <div className="max-w-lg space-y-6">
            <h2 className="font-serif text-3xl md:text-4xl text-brand-accent">A blank canvas.</h2>
            <p className="text-brand-subtext leading-relaxed">
              Begin building your digital wardrobe by uploading a piece you own. Our AI will automatically categorize it and extract its visual signature.
            </p>
            <div className="pt-4">
              <Link to="/add" className="btn-primary bg-brand-text">Start Building</Link>
            </div>
          </div>
        </section>
      )}

      {/* ── Features ── */}
      <section className="grid md:grid-cols-2 gap-12 md:gap-24">
        
        {/* Style Studio */}
        <div className="flex flex-col gap-6 items-start group">
          <div className="w-14 h-14 bg-[#EAE7DF] rounded-full flex items-center justify-center text-brand-text transition-transform duration-500 group-hover:scale-110">
            <Sparkles strokeWidth={1.5} />
          </div>
          <div>
            <h3 className="font-serif text-2xl mb-3 group-hover:text-brand-accent transition-colors">Style Studio</h3>
            <p className="text-brand-subtext leading-relaxed max-w-sm mb-8">
              Create looks from what you already own. Select items and let the AI suggest compatible combinations.
            </p>
            <Link to="/outfit" className="text-sm uppercase tracking-widest font-semibold flex items-center gap-2 group-hover:gap-3 transition-all">
              Generate Outfit <ArrowRight size={14} />
            </Link>
          </div>
        </div>

        {/* Missing Item */}
        <div className="flex flex-col gap-6 items-start group">
          <div className="w-14 h-14 bg-[#EAE7DF] rounded-full flex items-center justify-center text-brand-text transition-transform duration-500 group-hover:scale-110">
            <Search strokeWidth={1.5} />
          </div>
          <div>
            <h3 className="font-serif text-2xl mb-3 group-hover:text-brand-accent transition-colors">Missing Something?</h3>
            <p className="text-brand-subtext leading-relaxed max-w-sm mb-8">
              Complete your look. Build a partial outfit and discover exactly what piece pulls it all together.
            </p>
            <Link to="/missing-item" className="text-sm uppercase tracking-widest font-semibold flex items-center gap-2 group-hover:gap-3 transition-all">
              Find Missing Item <ArrowRight size={14} />
            </Link>
          </div>
        </div>

      </section>

    </div>
  );
}
