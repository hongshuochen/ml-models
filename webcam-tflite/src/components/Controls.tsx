'use client';

interface Props {
  intervalMs: number;
  onIntervalChange: (ms: number) => void;
  devices: MediaDeviceInfo[];
  activeDeviceId: string | null;
  onDeviceChange: (id: string) => void;
  disabled?: boolean;
}

/** Floating settings cluster: capture interval + camera picker. */
export function Controls({
  intervalMs,
  onIntervalChange,
  devices,
  activeDeviceId,
  onDeviceChange,
  disabled,
}: Props) {
  return (
    <div className="settings glass hud">
      <div className="set-row">
        <div className="set-head">
          <span className="set-label">Interval</span>
          <span className="set-val">{intervalMs === 0 ? 'max' : `${intervalMs} ms`}</span>
        </div>
        <input
          type="range"
          min={0}
          max={1000}
          step={50}
          value={intervalMs}
          disabled={disabled}
          onChange={(e) => onIntervalChange(Number(e.target.value))}
          aria-label="Capture interval"
        />
      </div>

      {devices.length > 1 && (
        <div className="set-row">
          <span className="set-label">Camera</span>
          <select
            className="set-select"
            value={activeDeviceId ?? ''}
            disabled={disabled}
            onChange={(e) => onDeviceChange(e.target.value)}
            aria-label="Camera"
          >
            {devices.map((d, i) => (
              <option key={d.deviceId} value={d.deviceId}>
                {d.label || `Camera ${i + 1}`}
              </option>
            ))}
          </select>
        </div>
      )}
    </div>
  );
}
