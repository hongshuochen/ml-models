'use client';

interface Props {
  message: string;
  onRetry?: () => void;
}

export function ErrorBanner({ message, onRetry }: Props) {
  if (!message) return null;
  return (
    <div className="banner" role="alert">
      <span>{message}</span>
      {onRetry && (
        <button className="btn-small" onClick={onRetry}>
          Retry
        </button>
      )}
    </div>
  );
}
