/** src/components/ui/EmptyState.jsx */
import { Link } from 'react-router-dom';

export default function EmptyState({
  icon: Icon,
  title,
  description,
  actionLabel,
  actionTo,
  onAction,
}) {
  return (
    <div className="flex flex-col items-center text-center py-20 px-6 animate-fade-in">
      {Icon && (
        <div className="w-14 h-14 rounded-full bg-stone-100 flex items-center justify-center mb-5">
          <Icon size={24} className="text-stone-400" strokeWidth={1.5} />
        </div>
      )}
      <h3 className="text-base font-semibold text-stone-800 mb-2">{title}</h3>
      {description && (
        <p className="text-sm text-stone-500 max-w-xs leading-relaxed mb-6">{description}</p>
      )}
      {actionLabel && actionTo && (
        <Link to={actionTo} className="btn-primary">
          {actionLabel}
        </Link>
      )}
      {actionLabel && onAction && (
        <button onClick={onAction} className="btn-primary">
          {actionLabel}
        </button>
      )}
    </div>
  );
}
