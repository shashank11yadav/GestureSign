import { REPO_URL } from '../links';

export function SiteHeader() {
  return (
    <header className="border-b border-line bg-paper/80 backdrop-blur">
      <div className="mx-auto flex h-14 max-w-6xl items-center justify-between gap-3 px-4">
        <a href="#main" className="flex min-w-0 items-center gap-2 font-semibold tracking-tight">
          <img src={`${import.meta.env.BASE_URL}icon.svg`} alt="" width={26} height={26} />
          <span>GestureSign</span>
          <span className="hidden rounded-full border border-line px-2 py-0.5 text-xs font-medium text-ink-muted sm:inline">
            Live demo
          </span>
        </a>
        <nav aria-label="Main" className="flex items-center gap-1 text-sm whitespace-nowrap">
          <a href="#how-it-works" className="rounded-md px-2.5 py-2 hover:bg-surface-muted">
            How it works
          </a>
          <a href={REPO_URL} className="rounded-md px-2.5 py-2 hover:bg-surface-muted">
            GitHub
          </a>
        </nav>
      </div>
    </header>
  );
}
