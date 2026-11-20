// Tabs.tsx — segmented tabs with a sliding active indicator (the A2 look).
// The indicator is one absolutely-positioned bar moved to the active tab's
// measured left/width, so tabs of any width work. Arrow keys move between tabs.

import React, { useLayoutEffect, useRef, useState } from 'react';

export interface TabItem<T extends string> {
  value: T;
  label: React.ReactNode;
  disabled?: boolean;
}

export interface TabsProps<T extends string> {
  items: Array<TabItem<T>>;
  value: T;
  onChange: (value: T) => void;
  className?: string;
  'aria-label'?: string;
}

export function Tabs<T extends string>({ items, value, onChange, className = '', 'aria-label': ariaLabel }: TabsProps<T>) {
  const listRef = useRef<HTMLDivElement>(null);
  const [box, setBox] = useState<{ left: number; width: number } | null>(null);

  useLayoutEffect(() => {
    const list = listRef.current;
    if (!list) return;
    const measure = () => {
      const el = list.querySelector<HTMLElement>('[aria-selected="true"]');
      setBox(el ? { left: el.offsetLeft, width: el.offsetWidth } : null);
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(list);
    return () => ro.disconnect();
  }, [value, items]);

  const onKeyDown = (e: React.KeyboardEvent) => {
    const dir = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0;
    if (!dir) return;
    e.preventDefault();
    const enabled = items.filter(i => !i.disabled);
    const next = enabled[(enabled.findIndex(i => i.value === value) + dir + enabled.length) % enabled.length];
    if (next) {
      onChange(next.value);
      listRef.current?.querySelector<HTMLElement>(`[data-tab="${next.value}"]`)?.focus();
    }
  };

  return (
    <div ref={listRef} role="tablist" aria-label={ariaLabel} onKeyDown={onKeyDown} className={`kit-tabs ${className}`}>
      {box && <span className="kit-tab-indicator" style={{ left: 0, width: box.width, transform: `translateX(${box.left}px)` }} aria-hidden="true" />}
      {items.map(i => (
        <button
          key={i.value}
          type="button"
          role="tab"
          data-tab={i.value}
          aria-selected={i.value === value}
          tabIndex={i.value === value ? 0 : -1}
          disabled={i.disabled}
          onClick={() => onChange(i.value)}
          className="kit-tab kit-focus disabled:cursor-not-allowed disabled:opacity-40"
        >
          {i.label}
        </button>
      ))}
    </div>
  );
}
