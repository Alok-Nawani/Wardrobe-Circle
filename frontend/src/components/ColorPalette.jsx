/**
 * src/components/ColorPalette.jsx
 * ================================
 * Displays dominant color + palette swatches from real API response.
 */

export function ColorSwatch({ hex, label, size = 'md' }) {
  const sizes = {
    sm: 'w-6 h-6',
    md: 'w-8 h-8',
    lg: 'w-10 h-10',
  };
  return (
    <div className="flex flex-col items-center gap-1" title={label || hex}>
      <div
        className={`${sizes[size]} rounded-full border border-stone-200 shadow-sm shrink-0`}
        style={{ backgroundColor: hex }}
        aria-label={label || hex}
      />
      {label && <span className="text-[10px] text-stone-400 max-w-[60px] truncate">{label}</span>}
    </div>
  );
}

export function ColorPalette({ dominantHex, dominantColor, palette = [], distribution = [] }) {
  return (
    <div className="space-y-3">
      {/* Dominant */}
      <div className="flex items-center gap-3">
        <div
          className="w-10 h-10 rounded-lg border border-stone-200 shadow-sm shrink-0"
          style={{ backgroundColor: dominantHex }}
          aria-label={`Dominant color: ${dominantColor}`}
        />
        <div>
          <p className="text-sm font-medium text-stone-800 capitalize">{dominantColor}</p>
          <p className="text-xs text-stone-400 font-mono">{dominantHex}</p>
        </div>
      </div>

      {/* Palette */}
      {palette.length > 0 && (
        <div>
          <p className="label mb-2">Color Palette</p>
          <div className="flex gap-2 flex-wrap">
            {palette.map((hex, i) => {
              const dist = distribution[i];
              return (
                <div key={hex + i} className="flex flex-col items-center gap-1">
                  <div
                    className="w-8 h-8 rounded-md border border-stone-100 shadow-sm"
                    style={{ backgroundColor: hex }}
                    title={hex}
                    aria-label={hex}
                  />
                  {dist && (
                    <span className="text-[10px] text-stone-400">
                      {Math.round(dist.percentage)}%
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
