/** src/components/ui/ErrorState.jsx */
import { AlertTriangle, RefreshCw } from 'lucide-react';

export default function ErrorState({ message, onRetry }) {
  return (
    <div className="flex flex-col items-center text-center py-16 px-6 animate-fade-in">
      <div className="w-12 h-12 rounded-full bg-red-50 flex items-center justify-center mb-4">
        <AlertTriangle size={20} className="text-red-400" />
      </div>
      <h3 className="text-sm font-semibold text-stone-800 mb-1">Something went wrong</h3>
      <p className="text-sm text-stone-500 max-w-sm leading-relaxed">{message}</p>
      {onRetry && (
        <button onClick={onRetry} className="mt-5 btn-secondary gap-2">
          <RefreshCw size={14} />
          Try again
        </button>
      )}
    </div>
  );
}
