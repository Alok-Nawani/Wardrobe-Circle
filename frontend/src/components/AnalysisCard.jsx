/**
 * src/components/AnalysisCard.jsx
 * ================================
 * Displays real Phase 1 + Phase 2 analysis results in an editorial typographic style.
 */

function AttributeList({ attributes }) {
  if (!attributes) return null;
  const source = attributes.source || '';
  const hasData = source !== 'not_in_dataset' &&
    (attributes.gender || attributes.season || attributes.base_colour ||
     attributes.article_type || attributes.product_name);

  if (!hasData) {
    return (
      <p className="text-sm text-brand-subtext italic">
        Additional catalog attributes are unavailable for this piece.
      </p>
    );
  }

  const rows = [
    { label: 'Product',  value: attributes.product_name },
    { label: 'Type',     value: attributes.article_type },
    { label: 'Gender',   value: attributes.gender },
    { label: 'Season',   value: attributes.season },
    { label: 'Colour',   value: attributes.base_colour },
  ].filter(r => r.value);

  return (
    <div className="space-y-2">
      {rows.map(r => (
        <div key={r.label} className="flex gap-4 text-sm">
          <span className="text-brand-subtext w-24 shrink-0 uppercase tracking-widest text-xs font-medium pt-0.5">{r.label}</span>
          <span className="text-brand-text capitalize">{r.value}</span>
        </div>
      ))}
      <div className="pt-2">
        <span className="text-[10px] text-brand-subtext uppercase tracking-widest border border-brand-border px-2 py-1">
          {source === 'dataset_metadata' ? 'From Catalog' : source}
        </span>
      </div>
    </div>
  );
}

export default function AnalysisCard({ analysis, similar, saved }) {
  if (!analysis) return null;
  const { category, confidence, dominant_color, dominant_hex, color_palette, color_distribution, attributes } = analysis;

  return (
    <div className="flex flex-col gap-10 border-l border-brand-border pl-8 py-4 animate-slide-up">
      
      {/* Category */}
      <div className="space-y-2">
        <h4 className="label-sm">Identified As</h4>
        <div className="flex items-baseline gap-3">
          <span className="font-serif text-4xl text-brand-text">{category}</span>
          {confidence !== null && (
            <span className="text-brand-subtext text-sm uppercase tracking-widest">
              {Math.round(confidence * 100)}% Match
            </span>
          )}
        </div>
      </div>

      {/* Color */}
      <div className="space-y-4">
        <h4 className="label-sm">Color Signature</h4>
        <div className="flex items-center gap-4">
          {dominant_hex && (
            <div 
              className="w-12 h-12 rounded-full border border-brand-border shadow-sm"
              style={{ backgroundColor: dominant_hex }}
            />
          )}
          <span className="text-xl font-serif text-brand-text capitalize">{dominant_color || 'Unknown'}</span>
        </div>
        
        {color_palette && color_palette.length > 0 && (
          <div className="pt-2">
            <p className="text-[10px] text-brand-subtext uppercase tracking-widest mb-2">Extracted Palette</p>
            <div className="flex flex-wrap gap-2">
              {color_palette.map((hex, i) => {
                const dist = color_distribution ? color_distribution[i] : null;
                const pct = dist ? Math.round(dist * 100) : null;
                return (
                  <div key={i} className="flex items-center gap-2 bg-white px-2 py-1 border border-brand-border">
                    <div className="w-3 h-3 rounded-full border border-brand-border/50" style={{ backgroundColor: hex }} />
                    <span className="text-xs text-brand-subtext font-mono uppercase">{hex}</span>
                    {pct !== null && <span className="text-[10px] text-brand-subtext/70">{pct}%</span>}
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* Attributes */}
      <div className="space-y-3">
        <h4 className="label-sm">Attributes</h4>
        <AttributeList attributes={attributes} />
      </div>

      {/* Similar */}
      {similar && similar.results && similar.results.length > 0 && (
        <div className="space-y-4 pt-4 border-t border-brand-border">
          <h4 className="label-sm">Visually Similar</h4>
          <div className="flex gap-4 overflow-x-auto pb-4 no-scrollbar">
            {similar.results.map((r, i) => (
              <div key={i} className="shrink-0 w-24 group cursor-pointer">
                <div className="clothing-image-container aspect-[3/4] mb-2">
                  <div className="w-full h-full flex items-center justify-center text-[10px] text-brand-subtext font-mono uppercase bg-white border border-brand-border p-2 text-center group-hover:border-brand-accent transition-colors">
                    {r.image_name}
                  </div>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-[10px] uppercase tracking-widest text-brand-text">{r.category}</span>
                  <span className="text-[10px] text-brand-subtext">{Math.round(r.similarity * 100)}%</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {saved && (
        <div className="pt-4 border-t border-brand-border">
          <span className="text-brand-accent uppercase tracking-widest text-xs font-semibold flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-brand-accent" />
            Saved to Collection
          </span>
        </div>
      )}
    </div>
  );
}
