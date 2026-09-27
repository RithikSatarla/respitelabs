export default function Nav() {
  const links = [
    ["Product", "/#product"],
    ["Demo", "/play/"],
    ["How it works", "/#story"],
    ["Results", "/#measured"],
    ["Deck", "/pitch/"],
  ];
  return (
    <header className="sticky top-0 z-40 border-b border-rule bg-paper/92 backdrop-blur">
      <nav className="mx-auto flex max-w-[1240px] items-center justify-between gap-4 px-5 py-3.5 sm:px-8">
        <a href="/" className="flex items-center gap-2">
          <span className="h-3.5 w-3.5 bg-accent" />
          <span className="text-[15px] font-extrabold tracking-[-0.02em]">kintrace</span>
        </a>
        <ul className="hidden items-center gap-7 md:flex">
          {links.map(([label, href]) => (
            <li key={label}>
              <a href={href} className="text-[13px] text-soft transition hover:text-ink">
                {label}
              </a>
            </li>
          ))}
        </ul>
        <a
          href="/play/"
          className="bg-accent px-4 py-2.5 text-[13px] font-bold text-ink transition hover:brightness-95"
        >
          Try it live
        </a>
      </nav>
    </header>
  );
}
