import type { ButtonHTMLAttributes, ReactNode } from 'react';

type Variant = 'primary' | 'secondary' | 'ghost' | 'danger' | 'onStage';

const VARIANTS: Record<Variant, string> = {
  primary:
    'bg-accent text-accent-ink shadow-sm hover:brightness-110 active:brightness-95 disabled:opacity-60',
  secondary: 'border border-line bg-surface text-ink hover:bg-surface-muted disabled:opacity-60',
  ghost: 'text-ink hover:bg-surface-muted disabled:opacity-50',
  danger: 'border border-line bg-surface text-danger hover:bg-danger-soft disabled:opacity-60',
  onStage:
    'border border-white/20 bg-white/10 text-white backdrop-blur hover:bg-white/20 disabled:opacity-60',
};

export function Button({
  variant = 'secondary',
  size = 'md',
  className = '',
  children,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: Variant;
  size?: 'sm' | 'md' | 'lg';
  children: ReactNode;
}) {
  const sizes = {
    sm: 'h-8 gap-1.5 px-3 text-sm',
    md: 'h-10 gap-2 px-4 text-sm',
    lg: 'h-12 gap-2 px-5 text-base',
  };
  return (
    <button
      type="button"
      className={`inline-flex shrink-0 cursor-pointer items-center justify-center rounded-lg font-medium whitespace-nowrap transition-colors disabled:cursor-not-allowed ${sizes[size]} ${VARIANTS[variant]} ${className}`}
      {...rest}
    >
      {children}
    </button>
  );
}

export function Card({ className = '', children }: { className?: string; children: ReactNode }) {
  return (
    <div className={`rounded-2xl border border-line bg-surface shadow-card ${className}`}>
      {children}
    </div>
  );
}

export function Percent({ value }: { value: number }) {
  return <span className="tabular-nums">{Math.round(value * 100)}%</span>;
}
