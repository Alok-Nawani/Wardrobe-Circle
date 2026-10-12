/**
 * src/components/ModelNote.jsx
 * ============================
 * Displays verbatim notes from the backend model in an editorial style.
 */

export default function ModelNote({ note }) {
  if (!note) return null;

  return (
    <div className="inline-flex flex-col items-center text-center gap-2 max-w-lg mx-auto">
      <span className="text-[10px] uppercase tracking-widest text-brand-subtext font-semibold">AI Curated</span>
      <p className="font-serif text-brand-text italic text-lg leading-relaxed">
        "{note}"
      </p>
    </div>
  );
}
