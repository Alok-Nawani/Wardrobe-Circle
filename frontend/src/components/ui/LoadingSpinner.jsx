/** src/components/ui/LoadingSpinner.jsx */
export function LoadingSpinner({ size = 20, className = '' }) {
  return (
    <svg
      className={`animate-spin text-stone-400 ${className}`}
      width={size} height={size}
      viewBox="0 0 24 24" fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-label="Loading"
    >
      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" strokeOpacity="0.25" />
      <path
        d="M12 2a10 10 0 0 1 10 10"
        stroke="currentColor" strokeWidth="3"
        strokeLinecap="round"
      />
    </svg>
  );
}

/** Full-page loading state */
export function PageLoading({ message = 'Loading…' }) {
  return (
    <div className="flex flex-col items-center justify-center py-24 gap-4 animate-fade-in">
      <LoadingSpinner size={32} className="text-stone-400" />
      <p className="text-sm text-stone-500">{message}</p>
    </div>
  );
}

/** Inline loading bar used during AI analysis */
export function AnalysisLoading({ message = 'Analyzing…' }) {
  return (
    <div className="flex flex-col items-center gap-3 py-10 animate-fade-in">
      <div className="relative w-48 h-1 bg-stone-100 rounded-full overflow-hidden">
        <div className="absolute left-0 top-0 h-full w-1/2 bg-stone-700 rounded-full animate-[shimmer_1.5s_ease-in-out_infinite]" />
      </div>
      <p className="text-sm text-stone-500">{message}</p>
    </div>
  );
}
