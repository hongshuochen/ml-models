'use client';

import { MODELS } from '@/lib/models';

interface Props {
  value: string;
  onChange: (id: string) => void;
  disabled?: boolean;
}

/** Minimal segmented control to switch between the bundled models. */
export function ModelSelector({ value, onChange, disabled }: Props) {
  return (
    <div className="segmented" role="tablist" aria-label="Model">
      {MODELS.map((m) => (
        <button
          key={m.id}
          role="tab"
          aria-selected={value === m.id}
          className={`seg ${value === m.id ? 'seg-on' : ''}`}
          disabled={disabled}
          onClick={() => onChange(m.id)}
        >
          {m.shortName}
        </button>
      ))}
    </div>
  );
}
