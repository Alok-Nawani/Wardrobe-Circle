/**
 * src/components/ScoreBadge.jsx
 * =============================
 * Displays compatibility score in a minimal typographic style.
 */

export default function ScoreBadge({ score }) {
  const pct = Math.round(score * 100);
  return (
    <div className="flex flex-col items-end">
      <span className="text-[10px] text-brand-subtext uppercase tracking-widest">Compat.</span>
      <span className="font-serif text-xl text-brand-accent">{pct}%</span>
    </div>
  );
}
